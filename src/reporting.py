import math
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from fastapi import HTTPException
import optuna
from optuna.trial import TrialState

from src.db_manager import get_db_session
from src.schema import TrialResult, AgentReasoningLog, StudyStatus, TrialLease
from src.hpo_config import load_hpo_config
from src.metrics import get_score, loss_objective_index, score_objective_index
from src.hpo_coordinator import compute_health_tier, write_ide_status_file, backfill_review_outcomes
from src.leases import _lease_is_owned
from src.pruning import _epoch_composite_score, _pruning_peer_trials
from src.suggest import load_study


class ReportEpochRequest(BaseModel):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    score: float
    loss: float
    gpu_memory: Optional[float] = 0.0
    speed_ips: Optional[float] = 0.0
    score_eval_fixed: Optional[float] = None
    loss_eval_fixed: Optional[float] = None


class CompleteTrialRequest(BaseModel):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    score: float
    loss: float
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
            raise HTTPException(status_code=404, detail=f"Trial ID {req.trial_id} not found.")

        # Idempotency check: if already completed, pruned, or failed, return the existing status
        # (BEFORE the lease check, so post-prune/idempotent retries don't 403 once the lease is gone).
        if trial_obj.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
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
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_dice", final_score)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_bce", final_loss)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "latest_epoch", req.epoch)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_memory", req.gpu_memory)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "speed_ips", req.speed_ips)

        hpo_config = load_hpo_config(req.study_name)
        ev = hpo_config.get("eval_protocol", {})
        if final_score_fixed is not None:
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_dice_attr", "dice_eval_fixed"), final_score_fixed
            )
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, "score_eval_fixed", final_score_fixed
            )
        if final_loss_fixed is not None:
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_bce_attr", "bce_eval_fixed"), final_loss_fixed
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
            "dice": final_score,
            "bce": final_loss
        }
        if final_score_fixed is not None:
            epoch_entry["score_eval_fixed"] = final_score_fixed
            epoch_entry["dice_eval_fixed"] = final_score_fixed
        if final_loss_fixed is not None:
            epoch_entry["loss_eval_fixed"] = final_loss_fixed
            epoch_entry["bce_eval_fixed"] = final_loss_fixed
        history.append(epoch_entry)
        study._storage.set_trial_user_attr(trial_obj._trial_id, "history", history)

        prune_score = final_score
        prune_loss = final_loss
        if ev.get("enabled") and ev.get("use_fixed_metric_for_pruning") and final_score_fixed is not None:
            prune_score = final_score_fixed
            prune_loss = final_loss_fixed if final_loss_fixed is not None else final_loss
        
        composite_score = _epoch_composite_score(study, trial_obj, req.epoch, ev)
        if composite_score is None:
            composite_score = prune_score - prune_loss

        prune_min_epoch = int(ev.get("prune_min_epoch", 5))

        if len(study.directions) == 1:
            study._storage.set_trial_intermediate_value(trial_obj._trial_id, req.epoch, prune_score)
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
                    session.query(TrialLease).filter_by(trial_id=req.trial_id).delete()
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
        if trial_obj.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            if not _lease_is_owned(req.study_name, req.trial_id, req.worker_id):
                raise HTTPException(
                    status_code=403,
                    detail="Trial is not leased to this worker_id; refusing to complete.",
                )

        final_score = req.score
        final_loss = req.loss
        final_score_fixed = req.score_eval_fixed
        final_loss_fixed = req.loss_eval_fixed

        hpo_config = load_hpo_config(req.study_name)
        ev = hpo_config.get("eval_protocol", {})

        # Only set Optuna user attrs if the trial is not finished yet in Optuna.
        # Otherwise, Optuna will raise UpdateFinishedTrialError.
        if trial_obj.state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            try:
                study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_memory", req.gpu_memory)
                study._storage.set_trial_user_attr(trial_obj._trial_id, "speed_ips", req.speed_ips)
                if req.gpu_model:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "gpu_model", req.gpu_model)
                if req.max_vram_gb is not None:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "max_vram_gb", req.max_vram_gb)
                if req.oom_triggered is not None:
                    study._storage.set_trial_user_attr(trial_obj._trial_id, "oom_triggered", req.oom_triggered)

                if final_score_fixed is not None:
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, ev.get("fixed_dice_attr", "dice_eval_fixed"), final_score_fixed
                    )
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, "score_eval_fixed", final_score_fixed
                    )
                if final_loss_fixed is not None:
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, ev.get("fixed_bce_attr", "bce_eval_fixed"), final_loss_fixed
                    )
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, "loss_eval_fixed", final_loss_fixed
                    )
            except optuna.exceptions.UpdateFinishedTrialError:
                # Log warning and proceed safely
                print(f"Warning: Trial #{trial_obj.number} was already finalized in Optuna. Skipping setting user attrs.")


        # Check if trial is already finished in Optuna
        if trial_obj.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            pass
        else:
            if t_state == TrialState.COMPLETE:
                if len(study.directions) > 1:
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
                        study.tell(trial_obj.number, final_loss)
                    else:
                        study.tell(trial_obj.number, final_score)
            else:
                try:
                    study.tell(trial_obj.number, state=t_state)
                except Exception as tell_err:
                    print(f"Could not tell state to Optuna: {tell_err}")

        with get_db_session() as session:
            detected_tag = None
            if math.isnan(final_score) or math.isnan(final_loss):
                detected_tag = "NAN_LOSS"
            elif math.isinf(final_score) or math.isinf(final_loss):
                detected_tag = "INF_GRADIENT"
            elif req.oom_triggered:
                detected_tag = "OOM"
            elif req.epoch <= 1 and final_score <= 0:
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
                failure_tag=detected_tag
            )
            metric.set_history(req.history)
            session.merge(metric)
            
            # Delete trial lease
            session.query(TrialLease).filter_by(trial_id=req.trial_id).delete()
            session.commit()

        try:
            prior_trials = [t for t in study.trials if t.number < trial_obj.number and t.state == TrialState.COMPLETE]
            best_prior_score = 0.0
            if prior_trials:
                scores = [get_score(t, study) for t in prior_trials]
                scores = [s for s in scores if s is not None]
                best_prior_score = max(scores) if scores else 0.0

            actual_improvement = final_score - best_prior_score
            with get_db_session() as session:
                reasoning_log = session.query(AgentReasoningLog).filter_by(trial_id=req.trial_id).first()
                if reasoning_log:
                    reasoning_log.actual_score_improvement = actual_improvement
                    session.commit()
        except Exception as reas_err:
            print(f"Error updating reasoning logs: {reas_err}")
            
        # Compute health tier and update study status
        try:
            health_tier, health_reason = compute_health_tier(study, req.study_name)
            with get_db_session() as session:
                status = session.query(StudyStatus).filter_by(study_name=req.study_name).first()
                if not status:
                    status = StudyStatus(study_name=req.study_name)
                    session.add(status)
                status.health_tier = health_tier
                status.health_reason = health_reason

            write_ide_status_file(req.study_name, health_tier, health_reason, study)
        except Exception as err:
            print(f"Error updating coordinator health status: {err}")

        try:
            backfill_review_outcomes(req.study_name)
        except Exception as bf_err:
            print(f"Error backfilling review outcomes: {bf_err}")

        # Fetch completed scores for sparkline
        completed_scores = []
        for t in study.trials:
            if t.state == TrialState.COMPLETE:
                s = get_score(t, study)
                if s is not None:
                    completed_scores.append(s)
        best_score = max(completed_scores) if completed_scores else 0.0

        return {
            "success": True,
            "completed_scores": completed_scores,
            "best_score": best_score,
            "completed_dices": completed_scores,
            "best_dice": best_score,
            "trial_number": trial_obj.number
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
