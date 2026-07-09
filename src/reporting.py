import math
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, model_validator
from fastapi import HTTPException
import optuna
from optuna.trial import TrialState

from src.db_manager import get_db_session, get_or_create_study_status
from src.schema import TrialResult
from src.hpo_config import load_hpo_config
from src.metrics import get_score, get_loss, loss_objective_index, score_objective_index, TERMINAL_STATES, has_invalid_metrics
from src.health import compute_health_tier, write_ide_status_file
from src.leases import _lease_is_owned, delete_lease_by_trial_id
from src.pruning import _epoch_composite_score, _pruning_peer_trials
from src.suggest import load_study

class AtLeastOneMetricMixin(BaseModel):
    score: Optional[float] = None
    loss: Optional[float] = None

    @model_validator(mode='after')
    def check_at_least_one_metric(self):
        if self.score is None and self.loss is None:
            raise ValueError('At least one of score or loss must be provided')
        return self


class ReportEpochRequest(AtLeastOneMetricMixin):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    gpu_memory: Optional[float] = 0.0
    speed_ips: Optional[float] = 0.0
    score_eval_fixed: Optional[float] = None
    loss_eval_fixed: Optional[float] = None

class CompleteTrialRequest(AtLeastOneMetricMixin):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    weights_path: str
    history: List[Dict[str, Any]]
    state: Optional[str] = "COMPLETE"
    gpu_memory: Optional[float] = 0.0
    speed_ips: Optional[float] = 0.0
    score_eval_fixed: Optional[float] = None
    loss_eval_fixed: Optional[float] = None
    gpu_model: Optional[str] = None
    max_vram_gb: Optional[float] = None
    oom_triggered: Optional[bool] = None
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    cuda_version: Optional[str] = None
    pip_freeze: Optional[str] = None
    dataset_version: Optional[str] = None
    hostname: Optional[str] = None
    platform: Optional[str] = None


def check_trial_health(study, score: Optional[float], loss: Optional[float], history: list) -> tuple[str, str]:
    """Checks for metric warnings and returns (health_tier, health_reason).
    
    Always returns a string pair; never None. When validation rules are disabled
    this returns ("healthy", "") so callers never need to guard against None values.
    """
    from src.hpo_config import load_hpo_config
    config = load_hpo_config(study.study_name)
    rules = config.get("validation_rules", {})
    if not rules.get("enabled", False):
        return "healthy", ""

    reasons = []
    
    # Check 1: Score < score_min on a maximize objective
    score_idx = score_objective_index(study)
    score_min = rules.get("score_min")
    if score_idx is not None and score is not None and score_min is not None:
        if score < score_min:
            reasons.append(f"Score < {score_min} ({score:.4f}) on a maximize objective")
            
    # Check 2: Loss < loss_min on a minimize objective
    loss_idx = loss_objective_index(study)
    loss_min = rules.get("loss_min")
    if loss_idx is not None and loss is not None and loss_min is not None:
        if loss < loss_min:
            reasons.append(f"Loss < {loss_min} ({loss:.4f}) on a minimize objective")
            
    # Check 3: Score change > max_epoch_jump between consecutive epochs (Warmup Gate: epoch > 5)
    max_jump = rules.get("max_epoch_jump")
    if max_jump is not None and history and len(history) >= 2:
        sorted_hist = sorted(history, key=lambda h: h.get("epoch", 0))
        for i in range(1, len(sorted_hist)):
            epoch = sorted_hist[i].get("epoch", 0)
            if epoch <= 5:
                continue
            prev_s = sorted_hist[i-1].get("score")
            curr_s = sorted_hist[i].get("score")
            if prev_s is not None and curr_s is not None and prev_s != 0.0:
                change = abs(curr_s - prev_s) / abs(prev_s)
                if change > max_jump:
                    reasons.append(
                        f"Score change > {max_jump*100:.0f}% ({change*100:.1f}%) between epoch {sorted_hist[i-1].get('epoch')} and {sorted_hist[i].get('epoch')}"
                    )
                    break
                    
    if reasons:
        return "watch", "; ".join(reasons)
    return "healthy", ""


def handle_api_report_epoch(req: ReportEpochRequest):
    try:
        study = load_study(req.study_name)
        should_prune = False

        trial_obj = None
        for t in study.trials:
            if t._trial_id == req.trial_id:
                trial_obj = t
                break

        if not trial_obj:
            raise HTTPException(status_code=404, detail=f"Trial ID {req.trial_id} not found. Has the trial been reaped? Try calling suggest() for a new trial.")

        # Idempotency check: if already completed, pruned, or failed, return the existing status
        # (BEFORE the lease check, so post-prune/idempotent retries don't 403 once the lease is gone).
        if trial_obj.state in TERMINAL_STATES:
            return {
                "should_prune": trial_obj.state == TrialState.PRUNED,
                "prune_metric": "none",
                "composite_score": 0.0,
                "peer_count": 0,
            }

        # Lease ownership: only the worker holding this trial's lease may report progress.
        if not _lease_is_owned(req.study_name, req.trial_id, req.worker_id):
            raise HTTPException(
                status_code=403,
                detail="Trial is not leased to this worker_id; re-acquire it via /api/suggest_trial.",
            )

        final_score = req.score
        final_loss = req.loss
        final_score_fixed = req.score_eval_fixed
        final_loss_fixed = req.loss_eval_fixed

        # Save user attributes for real-time dashboard monitoring
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_score", final_score)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_loss", final_loss)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_epoch", req.epoch)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_memory", req.gpu_memory)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "speed_ips", req.speed_ips)

        hpo_config = load_hpo_config(req.study_name)
        ev = hpo_config.get("eval_protocol", {})
        if final_score_fixed is not None:
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_score_attr", "score_eval_fixed"), final_score_fixed
            )
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, "score_eval_fixed", final_score_fixed
            )
        if final_loss_fixed is not None:
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_loss_attr", "loss_eval_fixed"), final_loss_fixed
            )
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, "loss_eval_fixed", final_loss_fixed
            )

        history = list(trial_obj.user_attrs.get("history", []))
        history = [h for h in history if h.get("epoch") != req.epoch]
        epoch_entry = {
            "epoch": req.epoch,
            "score": final_score,
            "loss": final_loss,
        }
        if final_score_fixed is not None:
            epoch_entry["score_eval_fixed"] = final_score_fixed
        if final_loss_fixed is not None:
            epoch_entry["loss_eval_fixed"] = final_loss_fixed
        history.append(epoch_entry)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "history", history)

        # Check trial metrics health
        health_tier, health_reason = check_trial_health(study, final_score, final_loss, history)
        try:
            study._storage.set_trial_user_attr(trial_obj._trial_id, "health_tier", health_tier or "healthy")
            study._storage.set_trial_user_attr(trial_obj._trial_id, "health_reason", health_reason or "")
        except Exception as e:
            print(f"Error saving health user attrs: {e}")

        prune_score = final_score
        prune_loss = final_loss
        if ev.get("enabled") and ev.get("use_fixed_metric_for_pruning") and final_score_fixed is not None:
            prune_score = final_score_fixed
            prune_loss = final_loss_fixed if final_loss_fixed is not None else final_loss
        
        composite_score = _epoch_composite_score(study, trial_obj, req.epoch, ev)
        if composite_score is None:
            s_val = prune_score if prune_score is not None else 0.0
            l_val = prune_loss if prune_loss is not None else 0.0
            composite_score = s_val - l_val

        prune_min_epoch = int(ev.get("prune_min_epoch", 5))

        if len(study.directions) == 1:
            intermediate_val = prune_loss if study.directions[0] == optuna.study.StudyDirection.MINIMIZE else prune_score
            if intermediate_val is None:
                intermediate_val = 0.0
            study._storage.set_trial_intermediate_value(trial_obj._trial_id, req.epoch, intermediate_val)
            should_prune = study.pruner.prune(study, trial_obj)
        else:
            study._storage.set_trial_intermediate_value(trial_obj._trial_id, req.epoch, composite_score)
            if req.epoch >= prune_min_epoch:
                past_scores = []
                for peer in _pruning_peer_trials(study, trial_obj, hpo_config):
                    peer_score = _epoch_composite_score(study, peer, req.epoch, ev)
                    if peer_score is not None:
                        past_scores.append(peer_score)

                if len(past_scores) >= 3:
                    past_scores.sort()
                    median_val = past_scores[len(past_scores) // 2]
                    if composite_score < median_val:
                        should_prune = True

        if should_prune:
            try:
                study.tell(trial_obj.number, state=optuna.trial.TrialState.PRUNED)
                # Cleanup lease immediately
                with get_db_session() as session:
                    delete_lease_by_trial_id(session, req.trial_id)
                    session.commit()
            except Exception as tell_err:
                print(f"Could not tell pruned state to Optuna for trial #{trial_obj.number}: {tell_err}")

        return {
            "should_prune": should_prune,
            "prune_metric": "fixed_eval" if (
                ev.get("enabled") and ev.get("use_fixed_metric_for_pruning") and final_score_fixed is not None
            ) else "train",
            "composite_score": composite_score,
            "peer_count": len(_pruning_peer_trials(study, trial_obj, hpo_config)) if req.epoch >= prune_min_epoch else 0,
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def handle_api_complete_trial(req: CompleteTrialRequest):
    try:
        study = load_study(req.study_name)

        t_state = TrialState.COMPLETE
        if req.state == "FAIL":
            t_state = TrialState.FAIL
        elif req.state == "PRUNED":
            t_state = TrialState.PRUNED

        trial_obj = None
        for t in study.trials:
            if t._trial_id == req.trial_id:
                trial_obj = t
                break

        if not trial_obj:
            raise HTTPException(status_code=404, detail=f"Trial ID {req.trial_id} not found.")

        # Lease ownership: in-flight trials must be completed by their lease holder. Terminal
        # trials skip this (idempotent retries and post-prune completes, where the lease was
        # already deleted when the trial was pruned, must still record their final result).
        if trial_obj.state not in TERMINAL_STATES:
            if not _lease_is_owned(req.study_name, req.trial_id, req.worker_id):
                raise HTTPException(
                    status_code=403,
                    detail="Trial is not leased to this worker_id; refusing to complete.",
                )

        final_score = req.score
        final_loss = req.loss
        final_score_fixed = getattr(req, "score_eval_fixed", None)
        final_loss_fixed = getattr(req, "loss_eval_fixed", None)

        hpo_config = load_hpo_config(req.study_name)
        ev = hpo_config.get("eval_protocol", {})

        # Metric validation
        if t_state == TrialState.COMPLETE:
            # Reject completes that look like training never ran:
            # - both score and loss are exactly 0.0
            # - multi-objective study (single-objective zero score is a valid minimum outcome)
            # - empty epoch history OR no evidence of progress (epoch <= 0 / no weights_path)
            no_history = not req.history or len(req.history) == 0
            no_progress = req.epoch <= 0 or not req.weights_path or req.weights_path.strip() == ""
            if final_score == 0.0 and final_loss == 0.0 and len(study.directions) >= 2 and (no_history or no_progress):
                raise HTTPException(
                    status_code=400,
                    detail="Rejecting complete: trial reported 0.0 for both score and loss. Likely training did not run.",
                )
        
        # NaN/Inf rejection only for COMPLETE state — FAIL trials must pass through
        # so the failure-tagging and health-monitoring logic below can record them.
        if t_state == TrialState.COMPLETE:
            invalid_metric = has_invalid_metrics(score=final_score, loss=final_loss, score_eval_fixed=final_score_fixed, loss_eval_fixed=final_loss_fixed)
            if invalid_metric:
                raise HTTPException(
                    status_code=400,
                    detail=f"Rejecting complete: {invalid_metric} is NaN or Inf, which is invalid.",
                )

        # Check trial metrics health
        health_tier, health_reason = check_trial_health(study, final_score, final_loss, req.history)

        # Only set Optuna user attrs if the trial is not finished yet in Optuna.
        # Otherwise, Optuna will raise UpdateFinishedTrialError.
        if trial_obj.state not in TERMINAL_STATES:
            try:
                study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_memory", req.gpu_memory)
                study._storage.set_trial_user_attr(trial_obj._trial_id, "speed_ips", req.speed_ips)
                if req.gpu_model:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_model", req.gpu_model)
                if req.max_vram_gb is not None:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "max_vram_gb", req.max_vram_gb)
                if req.oom_triggered is not None:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "oom_triggered", req.oom_triggered)
                
                study._storage.set_trial_user_attr(trial_obj._trial_id, "health_tier", health_tier or "healthy")
                study._storage.set_trial_user_attr(trial_obj._trial_id, "health_reason", health_reason or "")

                if final_score_fixed is not None:
                    study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_score_attr", "score_eval_fixed"), final_score_fixed
                    )
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, "score_eval_fixed", final_score_fixed
                    )
                if final_loss_fixed is not None:
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, ev.get("fixed_loss_attr", "loss_eval_fixed"), final_loss_fixed
                    )
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, "loss_eval_fixed", final_loss_fixed
                    )
            except optuna.exceptions.UpdateFinishedTrialError:
                # Log warning and proceed safely
                print(f"Warning: Trial #{trial_obj.number} was already finalized in Optuna. Skipping setting user attrs.")


        # Check if trial is already finished in Optuna
        if trial_obj.state in TERMINAL_STATES:
            pass
        else:
            if t_state == TrialState.COMPLETE:
                if len(study.directions) > 1:
                    if final_score is None or final_loss is None:
                        raise HTTPException(status_code=400, detail="Both score and loss must be provided to complete a dual-objective study.")
                    loss_idx = loss_objective_index(study)
                    score_idx = score_objective_index(study)
                    values = [0.0] * len(study.directions)
                    if loss_idx is not None:
                        values[loss_idx] = final_loss
                    if score_idx is not None:
                        values[score_idx] = final_score
                    study.tell(trial_obj.number, values)
                else:
                    if study.directions[0] == optuna.study.StudyDirection.MINIMIZE:
                        if final_loss is None:
                            raise HTTPException(status_code=400, detail="Trial completion for a minimize study requires 'loss'.")
                        study.tell(trial_obj.number, final_loss)
                    else:
                        if final_score is None:
                            raise HTTPException(status_code=400, detail="Trial completion for a maximize study requires 'score'.")
                        study.tell(trial_obj.number, final_score)
            else:
                try:
                    study.tell(trial_obj.number, state=t_state)
                except Exception as tell_err:
                    print(f"Could not tell state to Optuna: {tell_err}")

        with get_db_session() as session:
            detected_tag = None
            if (final_score is not None and math.isnan(final_score)) or (final_loss is not None and math.isnan(final_loss)):
                detected_tag = "NaN / Diverged"
            elif (final_score is not None and math.isinf(final_score)) or (final_loss is not None and math.isinf(final_loss)):
                detected_tag = "INF_GRADIENT"
            elif req.oom_triggered:
                detected_tag = "OOM"
            elif req.epoch <= 1 and final_score is not None and final_score <= 0:
                detected_tag = "DIVERGED"

            metric = TrialResult(
                trial_id=req.trial_id,
                study_name=req.study_name,
                epoch_reached=req.epoch,
                primary_loss=final_loss,
                primary_score=final_score,
                weights_path=req.weights_path,
                gpu_model=req.gpu_model,
                max_vram_gb=req.max_vram_gb,
                oom_triggered=req.oom_triggered,
                failure_tag=detected_tag,
                worker_id=req.worker_id,
                git_commit=req.git_commit,
                dataset_version=req.dataset_version,
                health_tier=health_tier or "healthy",
                health_reason=health_reason or "",
            )
            metric.set_history(req.history)
            session.merge(metric)
            
            # Save environment info to SystemConfiguration under worker_env:{worker_id} once to avoid duplicates
            if req.worker_id:
                env_dict = {
                    "python_version": req.python_version,
                    "cuda_version": req.cuda_version,
                    "pip_freeze": req.pip_freeze,
                    "platform": req.platform,
                    "hostname": req.hostname,
                }
                # Clean None values
                env_dict = {k: v for k, v in env_dict.items() if v is not None}
                if env_dict:
                    import json
                    from src.schema import SystemConfiguration
                    session.merge(SystemConfiguration(
                        study_name=req.study_name,
                        config_key=f"worker_env:{req.worker_id}",
                        config_value=json.dumps(env_dict)
                    ))
            
            # Delete trial lease
            delete_lease_by_trial_id(session, req.trial_id)
            session.commit()

        is_minimize_only = len(study.directions) == 1 and study.directions[0] == optuna.study.StudyDirection.MINIMIZE


        # Compute health tier and update study status
        try:
            health_tier, health_reason = compute_health_tier(study, req.study_name)
            with get_db_session() as session:
                status = get_or_create_study_status(session, req.study_name)
                status.health_tier = health_tier
                status.health_reason = health_reason

            write_ide_status_file(req.study_name, health_tier, health_reason, study)
        except Exception as err:
            print(f"Error updating study health status: {err}")

        # Fetch completed scores for sparkline
        completed_scores = []
        for t in study.trials:
            if t.state == TrialState.COMPLETE:
                s = get_loss(t, study) if is_minimize_only else get_score(t, study)
                if s is not None:
                    completed_scores.append(s)
        
        if is_minimize_only:
            best_score = min(completed_scores) if completed_scores else 0.0
        else:
            best_score = max(completed_scores) if completed_scores else 0.0

        return {
            "success": True,
            "completed_scores": completed_scores,
            "best_score": best_score,
            "trial_number": trial_obj.number
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
