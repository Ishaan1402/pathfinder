import hmac
import json
import traceback
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import optuna
from optuna.trial import TrialState

from ..db_manager import get_db_session, get_or_create_study_status
from ..settings import settings
from ..schema import (
    StudyStatus,
    TrialResult,
    SystemConfiguration,
    CoordinatorMetric,
    SuggestMetric,
    InvalidProposal,
)
from ..hpo_config import (
    load_hpo_config,
    save_hpo_config,
    normalize_trial_params,
    param_display_name,
)
from ..metrics import score_objective_index, _trial_metric_snapshot
from ..search_space import (
    _migrate_search_space,
    load_search_space,
    _apply_search_space_patch,
    handle_api_get_search_space,
    handle_api_update_search_space,
)
from ..pruning import _effective_train_resolution
from ..suggest import (
    get_or_create_study,
    load_study,
    _enqueue_manual_trial,
)
from ..leases import (
    _reap_stale_running_trials,
    _reap_expired_leases,
)
from ..hpo_coordinator import (
    study_eval_insights as _study_eval_insights,
    pareto_trial_numbers_deploy_aware as _pareto_trial_numbers_deploy_aware,
    build_review_packet,
    save_study_review,
    get_recent_study_reviews,
    count_evaluated_trials,
    compute_review_heuristics,
    compute_statistical_confidence,
    validate_review_fields,
    mark_review_applied,
    flag_study_review,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    token: str


class InitFromManifestRequest(BaseModel):
    yaml: str


class AgentReviewRequest(BaseModel):
    study_name: str
    summary: str
    health_rating: Optional[int] = None  # 1-5
    policy_action: Optional[str] = "no_change"  # no_change | update_active_search_space | enqueue_one_manual_trial
    model_version: Optional[str] = "coordinator"
    prompt_strategy: Optional[str] = "coordinator_review"
    reasons: Optional[List[Dict[str, Any]]] = None
    search_space_patch: Optional[Dict[str, Any]] = None
    manual_trial: Optional[Dict[str, Any]] = None
    estimated_score_improvement: Optional[float] = None
    cited_best_trial: Optional[int] = None
    force: Optional[bool] = False


@router.post("/login")
def api_login(req: LoginRequest, request: Request):
    """Exchange the shared token for an httpOnly session cookie (dashboard login)."""
    secret_token = settings.secret_token
    if not secret_token:
        return JSONResponse({"success": True, "auth_required": False})
    if not hmac.compare_digest(req.token, secret_token):
        raise HTTPException(status_code=401, detail="Invalid token.")
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    is_https = request.url.scheme == "https" or forwarded_proto == "https"
    resp = JSONResponse({"success": True, "auth_required": True})
    resp.set_cookie(
        key="hpo_session",
        value=secret_token,
        httponly=True,
        samesite="lax",
        secure=is_https,
        max_age=7 * 24 * 3600,
        path="/",
    )
    return resp


@router.get("/hpo_config")
def api_get_hpo_config(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    return load_hpo_config(study_name)


@router.post("/hpo_config")
def api_save_hpo_config(config: Dict[str, Any], study_name: Optional[str] = None):
    if not study_name:
        study_name = config.get("study_name") or settings.study_name
    config_clean = {k: v for k, v in config.items() if k != "study_name"}
    save_hpo_config(config_clean, study_name)
    return {"success": True, "config": load_hpo_config(study_name)}


@router.get("/search_space")
def api_get_search_space(study_name: Optional[str] = None):
    return handle_api_get_search_space(study_name)


@router.post("/update_search_space")
def api_update_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    return handle_api_update_search_space(space, study_name)


@router.get("/study_health")
def api_get_study_health(study_name: str):
    try:
        study = get_or_create_study(study_name)
        with get_db_session() as session:
            _reap_stale_running_trials(study, study_name, session)
    except Exception as reap_err:
        print(f"study_health reap skipped for '{study_name}': {reap_err}")

    with get_db_session() as session:
        status = session.query(StudyStatus).filter_by(study_name=study_name).first()
        is_dismissed = False
        if status:
            try:
                study = get_or_create_study(study_name)
                trials_evaluated = count_evaluated_trials(study)
                if status.nudge_dismissed_trials == trials_evaluated:
                    is_dismissed = True
            except Exception as e:
                logger.warning(f"Failed to load study {study_name} when checking dismissal status: {e}")
            return {
                "study_name": study_name,
                "health_tier": status.health_tier,
                "health_reason": status.health_reason,
                "health_updated_at": status.health_updated_at.isoformat() if status.health_updated_at else None,
                "is_dismissed": is_dismissed
            }
        return {
            "study_name": study_name,
            "health_tier": "healthy",
            "health_reason": "No status found, defaulting to healthy.",
            "health_updated_at": None,
            "is_dismissed": False
        }


@router.get("/pending_changes")
def api_get_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    with get_db_session() as session:
        row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        if row:
            try:
                return {"proposed_changes": json.loads(row.config_value)}
            except Exception as e:
                return {"proposed_changes": None, "error": str(e)}
    return {"proposed_changes": None}


@router.post("/apply_pending_changes")
def api_apply_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    try:
        with get_db_session() as session:
            pending_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="pending_search_space"
            ).first()
            if not pending_row:
                raise HTTPException(status_code=400, detail="No pending changes found.")
            
            proposed = json.loads(pending_row.config_value)
            current = load_search_space(study_name)
            
            for key, new_val in proposed.items():
                if key not in current:
                    raise HTTPException(status_code=400, detail=f"Parameter {key} not in active search space.")
                
                p_type = current[key].get("type")
                if p_type == "categorical":
                    if "active" in new_val:
                        allowed = current[key].get("options", [])
                        invalid = [x for x in new_val["active"] if x not in allowed]
                        if invalid:
                            raise HTTPException(status_code=400, detail=f"Invalid active options for {key}: {invalid}")
                        if not new_val["active"]:
                            raise HTTPException(status_code=400, detail=f"Must keep at least one active option for {key}.")
                        current[key]["active"] = new_val["active"]
                else:
                    if "min" in new_val:
                        current[key]["min"] = float(new_val["min"])
                    if "max" in new_val:
                        current[key]["max"] = float(new_val["max"])
            
            session.merge(SystemConfiguration(
                study_name=study_name,
                config_key="active_search_space",
                config_value=json.dumps(_migrate_search_space(current))
            ))
            session.delete(pending_row)
            mark_review_applied(study_name)
            return {"success": True, "space": current}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply pending changes: {str(e)}")


@router.post("/discard_pending_changes")
def api_discard_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    try:
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="pending_search_space"
            ).first()
            if row:
                session.delete(row)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discard pending changes: {str(e)}")


@router.post("/validate_manifest")
def api_validate_manifest(req: InitFromManifestRequest):
    import yaml
    from ..manifest import validate_manifest
    try:
        data = yaml.safe_load(req.yaml)
    except Exception as e:
        return {"success": False, "errors": [f"Invalid YAML structure: {str(e)}"], "warnings": []}
        
    if not isinstance(data, dict):
        return {"success": False, "errors": ["Manifest root must be a dictionary"], "warnings": []}

    errors, warnings = validate_manifest(data)
    return {"success": len(errors) == 0, "errors": errors, "warnings": warnings}


@router.post("/init_from_manifest")
def api_init_from_manifest(req: InitFromManifestRequest, force: bool = False):
    import yaml
    from ..manifest import validate_manifest
    from ..onboarding import init_study_from_manifest_dict
    
    try:
        data = yaml.safe_load(req.yaml)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML structure: {str(e)}")
        
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Manifest root must be a dictionary")

    errors, warnings = validate_manifest(data)
    if errors:
        return {"success": False, "errors": errors, "warnings": warnings}

    try:
        result = init_study_from_manifest_dict(data, force=force)
        return {"success": True, "study_name": data["study_name"], "message": result, "warnings": warnings}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/studies")
def api_list_studies():
    try:
        summaries = optuna.get_all_study_summaries(storage=settings.database_url)
        return {"success": True, "studies": [s.study_name for s in summaries]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/study_setup")
def api_study_setup(study_name: str):
    try:
        with get_db_session() as session:
            context_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="project_context"
            ).first()
            hpo_config_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="hpo_config"
            ).first()
            context_val = context_row.config_value if context_row else None
            hpo_config_val = hpo_config_row.config_value if hpo_config_row else None
            
        context = json.loads(context_val) if context_val else {}
        hpo_config = json.loads(hpo_config_val) if hpo_config_val else {}
        
        is_reference = (study_name == "bridge_crack_study") and ("worker_entrypoint" not in context)
        
        return {
            "success": True,
            "study_name": study_name,
            "worker_entrypoint": context.get("worker_entrypoint"),
            "worker_env": context.get("worker_env"),
            "is_reference": is_reference,
            "manifest_metrics": hpo_config.get("manifest_metrics"),
            "colab_snippet": context.get("colab_snippet")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/study_details")
def api_study_details(study_name: str):
    try:
        study = get_or_create_study(study_name)

        with get_db_session() as session:
            _reap_expired_leases(study, study_name, session)

        with get_db_session() as session:
            metric_rows = session.query(TrialResult).filter_by(study_name=study_name).all()
            metrics_dict = {m.trial_id: m.get_history() for m in metric_rows}
            trial_result_map = {
                m.trial_id: {
                    "oom_triggered": m.oom_triggered,
                    "failure_tag": m.failure_tag,
                    "gpu_model": m.gpu_model,
                    "max_vram_gb": m.max_vram_gb,
                    "worker_id": m.worker_id,
                    "git_commit": m.git_commit,
                    "dataset_version": m.dataset_version,
                    "health_tier": m.health_tier,
                    "health_reason": m.health_reason,
                }
                for m in metric_rows
            }
            env_rows = session.query(SystemConfiguration).filter(
                SystemConfiguration.study_name == study_name,
                SystemConfiguration.config_key.like("worker_env:%")
            ).all()
            worker_envs = {}
            for r in env_rows:
                try:
                    w_id = r.config_key.replace("worker_env:", "")
                    worker_envs[w_id] = json.loads(r.config_value)
                except Exception as e:
                    logger.warning(f"Failed to parse worker_env {w_id}: {e}")

        hpo_config = load_hpo_config(study_name)
        space = load_search_space(study_name)
        ev = hpo_config.get("eval_protocol", {})
        dice_fixed_attr = ev.get("fixed_dice_attr", "dice_eval_fixed")
        bce_fixed_attr = ev.get("fixed_bce_attr", "bce_eval_fixed")
        train_param = ev.get("train_resolution_param", "resolution")

        trials_list = []
        for t in study.trials:
            gpu_memory = t.user_attrs.get("gpu_memory", None)
            speed_ips = t.user_attrs.get("speed_ips", None)

            history = t.user_attrs.get("history", [])
            if not history and t._trial_id in metrics_dict:
                history = metrics_dict[t._trial_id]

            metrics = _trial_metric_snapshot(t, history, dice_fixed_attr, bce_fixed_attr, study.directions)
            train_res = _effective_train_resolution(t, hpo_config, space)
            norm_params = normalize_trial_params(dict(t.params), hpo_config)
            if train_res is not None and train_param not in norm_params:
                norm_params[train_param] = train_res

            tr = trial_result_map.get(t._trial_id) or {}
            w_id = tr.get("worker_id")
            w_env = worker_envs.get(w_id, {}) if w_id else {}
            trials_list.append({
                "number": t.number,
                "trial_id": t._trial_id,
                "state": t.state.name,
                "params": norm_params,
                "params_display": {
                    param_display_name(k, hpo_config): v for k, v in norm_params.items()
                },
                "bce": metrics["bce"],
                "dice": metrics["dice"],
                "score": metrics["score"],
                "loss": metrics["loss"],
                "dice_eval_fixed": metrics["dice_eval_fixed"],
                "bce_eval_fixed": metrics["bce_eval_fixed"],
                "score_eval_fixed": metrics["score_eval_fixed"],
                "loss_eval_fixed": metrics["loss_eval_fixed"],
                "train_resolution": train_res,
                "latest_epoch": metrics["latest_epoch"],
                "gpu_memory": gpu_memory,
                "speed_ips": speed_ips,
                "history": history,
                "intermediate_values": t.intermediate_values,
                "oom_triggered": tr.get("oom_triggered") or t.user_attrs.get("oom_triggered", False),
                "failure_tag": tr.get("failure_tag"),
                "gpu_model": tr.get("gpu_model"),
                "max_vram_gb": tr.get("max_vram_gb"),
                "worker_id": w_id,
                "git_commit": tr.get("git_commit") or t.user_attrs.get("git_commit", None),
                "dataset_version": tr.get("dataset_version") or t.user_attrs.get("dataset_version", None),
                "health_tier": tr.get("health_tier") or t.user_attrs.get("health_tier", None),
                "health_reason": tr.get("health_reason") or t.user_attrs.get("health_reason", None),
                "hostname": w_env.get("hostname"),
                "platform": w_env.get("platform"),
                "python_version": w_env.get("python_version"),
                "cuda_version": w_env.get("cuda_version"),
                "pip_freeze": w_env.get("pip_freeze"),
            })
            
        running_count = sum(1 for t in study.trials if t.state == TrialState.RUNNING)
        
        pareto_trial_numbers = []
        if len(study.directions) > 1:
            pareto_trial_numbers = _pareto_trial_numbers_deploy_aware(study, hpo_config)
        
        insights = _study_eval_insights(study, hpo_config)
        review = compute_review_heuristics(study, insights, hpo_config, study_name)
        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

        return {
            "study_name": study_name,
            "running_workers": running_count,
            "trials": trials_list,
            "pareto_trials": pareto_trial_numbers,
            "study_directions": [d.name for d in study.directions],
            "hpo_config": hpo_config,
            "eval_insights": insights,
            "review": review,
            "statistical_confidence": compute_statistical_confidence(n_complete),
            "completed_count": n_complete,
            "past_reviews": get_recent_study_reviews(study_name, limit=10),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fanova")
def api_fanova(study_name: str):
    try:
        study = get_or_create_study(study_name)
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(complete_trials) < 2:
            return {"success": False, "message": "Need at least 2 completed trials for importance analysis"}
        config = load_hpo_config(study_name)
        from ..hpo_coordinator import get_fanova_importances
        display = get_fanova_importances(study, config)

        return {"success": True, "importances": display}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.get("/review_packet")
def api_review_packet(study_name: str):
    """Read-only context for the IDE coordinator: Pareto, fANOVA, eval insights, drift reasons."""
    try:
        return build_review_packet(study_name)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pareto_front")
def api_pareto_front(study_name: str):
    """Exposes Pareto front trials for export in the GUI dashboard."""
    try:
        study = get_or_create_study(study_name)
        if len(study.directions) < 2:
            return {"success": True, "pareto_front": []}
            
        hpo_config = load_hpo_config(study_name)
        ev = hpo_config.get("eval_protocol", {})
        dice_fixed_attr = ev.get("fixed_dice_attr", "dice_eval_fixed")
        bce_fixed_attr = ev.get("fixed_bce_attr", "bce_eval_fixed")
        
        with get_db_session() as session:
            metrics = session.query(TrialResult).filter_by(study_name=study_name).all()
            metrics_dict = {m.trial_id: m.get_history() for m in metrics}

        pareto_trials = []
        best_trials = study.best_trials
        for t in best_trials:
            history = t.user_attrs.get("history", [])
            if not history and t._trial_id in metrics_dict:
                history = metrics_dict[t._trial_id]
                
            metrics_vals = _trial_metric_snapshot(t, history, dice_fixed_attr, bce_fixed_attr, study.directions)
            norm_params = normalize_trial_params(dict(t.params), hpo_config)
            
            pareto_trials.append({
                "number": t.number,
                "trial_id": t._trial_id,
                "bce": metrics_vals["bce"],
                "dice": metrics_vals["dice"],
                "score": metrics_vals["score"],
                "loss": metrics_vals["loss"],
                "params": norm_params
            })
            
        return {"success": True, "pareto_front": pareto_trials}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dismiss_coordinator_nudge")
def api_dismiss_coordinator_nudge(study_name: str):
    """Dismisses the coordinator nudge for the current trial window by persisting it in SQLite."""
    try:
        study = get_or_create_study(study_name)
        trials_evaluated = count_evaluated_trials(study)
        
        with get_db_session() as session:
            status = get_or_create_study_status(session, study_name)
            status.nudge_dismissed_trials = trials_evaluated
            session.commit()
            
        return {"success": True, "dismissed_trials": trials_evaluated}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/study_cards")
def api_get_study_cards(study_name: Optional[str] = None):
    """Exposes generated study cards and their markdown content for dashboard retrieval."""
    try:
        return {"success": True, "cards": load_study_cards(study_name)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent_review")
def api_agent_review(req: AgentReviewRequest):
    """Persist a coordinator review. Idempotent per trial window unless force=True."""
    try:
        study = load_study(req.study_name)
        space = load_search_space(req.study_name)
        trials_evaluated = count_evaluated_trials(study)

        if req.manual_trial:
            from ..hpo_coordinator import _validate_manual_parameters
            val_res = _validate_manual_parameters(req.manual_trial, req.study_name)
            if not val_res["ok"]:
                with get_db_session() as session:
                    session.add(InvalidProposal(
                        study_name=req.study_name,
                        model_version=req.model_version or "coordinator",
                        prompt_strategy=req.prompt_strategy or "coordinator_review",
                        invalid_parameters=json.dumps(req.manual_trial),
                        validation_error=val_res["error"]
                    ))
                return {"success": False, "error": f"Invalid manual parameters: {val_res['error']}"}

        validation = validate_review_fields(req.estimated_score_improvement, req.cited_best_trial)
        if not validation["ok"]:
            return {"success": False, "error": "; ".join(validation["errors"])}
        result = save_study_review(
            req.study_name,
            req.summary,
            health_rating=req.health_rating,
            policy_action=req.policy_action or "no_change",
            model_version=req.model_version or "coordinator",
            prompt_strategy=req.prompt_strategy or "coordinator_review",
            reasons=req.reasons,
            trials_evaluated=trials_evaluated,
            estimated_score_improvement=req.estimated_score_improvement,
            cited_best_trial=req.cited_best_trial,
            force=bool(req.force),
        )

        applied = {}
        if not result.get("duplicate"):
            if req.search_space_patch:
                applied["search_space"] = _apply_search_space_patch(req.search_space_patch, space, req.study_name)
            if req.manual_trial:
                applied["manual_trial"] = _enqueue_manual_trial(study, req.manual_trial, space, req.summary)

        result["applied"] = applied
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flag_review")
def api_flag_review(review_id: int, flagged: bool = True):
    """Mark a coordinator review as low-quality (excluded from accuracy MAE)."""
    try:
        return flag_study_review(review_id, flagged=flagged)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tunnel_url")
def api_get_tunnel_url():
    """Returns the active remote broker URL if established."""
    try:
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name="_global", config_key="remote_broker_url"
            ).first()
            if not row:
                row = session.query(SystemConfiguration).filter_by(
                    study_name="_global", config_key="ngrok_tunnel_url"
                ).first()
            return {"success": True, "url": row.config_value if row else None}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/debug/config_audit")
def api_config_audit(study_name: Optional[str] = None):
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found.")
    if not study_name:
        study_name = settings.study_name
    
    report = {
        "study_name": study_name,
        "violations": [],
    }

    db_space = load_search_space(study_name)

    try:
        study = get_or_create_study(study_name)
        for t in study.trials:
            if t.state not in (TrialState.COMPLETE, TrialState.RUNNING):
                continue
            for param, val in t.params.items():
                if param in db_space:
                    cfg = db_space[param]
                    ptype = cfg.get("type")
                    if ptype == "categorical":
                        active = cfg.get("active", cfg.get("options", []))
                        if val not in active:
                            report["violations"].append({
                                "trial_number": t.number,
                                "trial_id": t._trial_id,
                                "param": param,
                                "value": val,
                                "reason": f"Value {val} not in active options {active}"
                            })
                    else:
                        lo = cfg.get("min")
                        hi = cfg.get("max")
                        if lo is not None and float(val) < float(lo):
                            report["violations"].append({
                                "trial_number": t.number,
                                "trial_id": t._trial_id,
                                "param": param,
                                "value": val,
                                "reason": f"Value {val} below active min {lo}"
                            })
                        if hi is not None and float(val) > float(hi):
                            report["violations"].append({
                                "trial_number": t.number,
                                "trial_id": t._trial_id,
                                "param": param,
                                "value": val,
                                "reason": f"Value {val} above active max {hi}"
                            })
    except Exception as e:
        report["trial_validation_error"] = str(e)
        
    return report


@router.get("/metrics/coordinator")
def api_metrics_coordinator(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    with get_db_session() as session:
        rows = session.query(CoordinatorMetric).filter_by(study_name=study_name).all()
        return {"success": True, "metrics": [r.to_dict() for r in rows]}


@router.get("/metrics/suggest")
def api_metrics_suggest(study_name: Optional[str] = None):
    if not study_name:
        study_name = settings.study_name
    with get_db_session() as session:
        rows = session.query(SuggestMetric).filter_by(study_name=study_name).all()
        return {"success": True, "metrics": [r.to_dict() for r in rows]}


@router.get("/mcp_info")
def api_mcp_info():
    return {
        "success": True,
        "mcp_server_name": "pathfinder",
        "active_study": settings.study_name,
        "mcp_tools": [
            "initialize_study",
            "get_study_data",
            "validate_search_space",
            "update_search_space",
            "delete_study",
            "generate_model_card",
            "submit_agent_review",
            "validate_integration",
            "get_study_cards",
            "validate_manifest",
            "init_from_manifest",
            "export_manifest",
        ]
    }
