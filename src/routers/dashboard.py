import hmac
import json
import traceback
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import optuna
from optuna.trial import TrialState

from ..db_manager import get_db_session, get_or_create_study_status
from ..settings import settings
from ..schema import (
    TrialResult,
    SystemConfiguration,
)
from ..hpo_config import (
    load_hpo_config,
    save_hpo_config,
    normalize_trial_params,
    param_display_name,
)
from ..metrics import _trial_metric_snapshot
from ..search_space import (
    load_search_space,
    handle_api_get_search_space,
    handle_api_update_search_space,
    _fixed_categorical_params,
)
from ..pruning import _effective_train_resolution
from ..suggest import load_study
from ..leases import (
    _reap_stale_running_trials,
    _reap_expired_leases,
)
from ..health import compute_health_tier, compute_statistical_confidence, count_evaluated_trials
from ..analytics import (
    study_eval_insights as _study_eval_insights,
    pareto_trial_numbers_deploy_aware as _pareto_trial_numbers_deploy_aware,
    build_study_packet,
    load_study_cards,
    get_fanova_importances,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class LoginRequest(BaseModel):
    token: str


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
        study = load_study(study_name)
        with get_db_session() as session:
            _reap_stale_running_trials(study, study_name, session)
    except Exception as reap_err:
        print(f"study_health reap skipped for '{study_name}': {reap_err}")

    study = load_study(study_name)
    health_tier, health_reason = compute_health_tier(study, study_name)
    trials_evaluated = count_evaluated_trials(study)

    with get_db_session() as session:
        status = get_or_create_study_status(session, study_name)
        status.health_tier = health_tier
        status.health_reason = health_reason

    return {
        "study_name": study_name,
        "health_tier": health_tier,
        "health_reason": health_reason,
        "trials_evaluated": trials_evaluated,
    }


@router.get("/studies")
def api_list_studies():
    try:
        summaries = optuna.get_all_study_summaries(storage=settings.database_url)
        return {"success": True, "studies": [s.study_name for s in summaries]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/study_details")
def api_study_details(study_name: str):
    try:
        study = load_study(study_name)

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
        score_fixed_attr = ev.get("fixed_score_attr", "score_eval_fixed")
        loss_fixed_attr = ev.get("fixed_loss_attr", "loss_eval_fixed")
        train_param = ev.get("train_resolution_param", "resolution")

        trials_list = []
        for t in study.trials:
            gpu_memory = t.user_attrs.get("gpu_memory", None)
            speed_ips = t.user_attrs.get("speed_ips", None)

            history = t.user_attrs.get("history", [])
            if not history and t._trial_id in metrics_dict:
                history = metrics_dict[t._trial_id]

            metrics = _trial_metric_snapshot(t, history, score_fixed_attr, loss_fixed_attr, study.directions)
            train_res = _effective_train_resolution(t, hpo_config, space)
            norm_params = normalize_trial_params(dict(t.params), hpo_config)
            for k, v in _fixed_categorical_params(space).items():
                if k not in norm_params:
                    norm_params[k] = v
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
                "score": metrics["score"],
                "loss": metrics["loss"],
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
        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        health_tier, health_reason = compute_health_tier(study, study_name)

        return {
            "study_name": study_name,
            "running_workers": running_count,
            "trials": trials_list,
            "pareto_trials": pareto_trial_numbers,
            "study_directions": [d.name for d in study.directions],
            "hpo_config": hpo_config,
            "eval_insights": insights,
            "health": {"tier": health_tier, "reason": health_reason},
            "statistical_confidence": compute_statistical_confidence(n_complete),
            "completed_count": n_complete,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fanova")
def api_fanova(study_name: str):
    try:
        study = load_study(study_name)
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(complete_trials) < 2:
            return {"success": False, "message": "Need at least 2 completed trials for importance analysis"}
        config = load_hpo_config(study_name)
        display = get_fanova_importances(study, config)

        return {"success": True, "importances": display}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.get("/study_packet")
def api_study_packet(study_name: str):
    """Read-only context for the IDE coordinator: Pareto, fANOVA, eval insights, drift reasons."""
    try:
        return build_study_packet(study_name)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pareto_front")
def api_pareto_front(study_name: str):
    """Exposes Pareto front trials for export in the GUI dashboard."""
    try:
        study = load_study(study_name)
        if len(study.directions) < 2:
            return {"success": True, "pareto_front": []}
            
        hpo_config = load_hpo_config(study_name)
        space = load_search_space(study_name)
        ev = hpo_config.get("eval_protocol", {})
        score_fixed_attr = ev.get("fixed_score_attr", "score_eval_fixed")
        loss_fixed_attr = ev.get("fixed_loss_attr", "loss_eval_fixed")
        
        with get_db_session() as session:
            metrics = session.query(TrialResult).filter_by(study_name=study_name).all()
            metrics_dict = {m.trial_id: m.get_history() for m in metrics}

        pareto_trials = []
        best_trials = study.best_trials
        for t in best_trials:
            history = t.user_attrs.get("history", [])
            if not history and t._trial_id in metrics_dict:
                history = metrics_dict[t._trial_id]
                
            metrics_vals = _trial_metric_snapshot(t, history, score_fixed_attr, loss_fixed_attr, study.directions)
            norm_params = normalize_trial_params(dict(t.params), hpo_config)
            for k, v in _fixed_categorical_params(space).items():
                if k not in norm_params:
                    norm_params[k] = v
            
            pareto_trials.append({
                "number": t.number,
                "trial_id": t._trial_id,
                "score": metrics_vals["score"],
                "loss": metrics_vals["loss"],
                "params": norm_params
            })
            
        return {"success": True, "pareto_front": pareto_trials}
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
        study = load_study(study_name)
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


@router.post("/quickstart_demo")
def api_quickstart_demo(request: Request):
    from ..onboarding import init_study_from_manifest_dict
    from threading import Thread
    from simulators.training_worker import run_training_worker

    DEMO_STUDY = "demo_segmentation_study"

    # If the demo study already exists and has trials, just redirect — no re-spawn needed
    try:
        existing = optuna.load_study(study_name=DEMO_STUDY, storage=settings.database_url)
        if len(existing.trials) > 0:
            return {"success": True, "study_name": DEMO_STUDY}
    except KeyError:
        pass  # Study doesn't exist yet — proceed

    DEMO_MANIFEST = {
        "study_name": DEMO_STUDY,
        "metrics": {
            "primary_score": "score",
            "objectives": [
                {"name": "loss", "direction": "minimize", "label": "Loss"},
                {"name": "score", "direction": "maximize", "label": "Score"},
            ],
        },
        "params": [
            {"name": "learning_rate", "type": "float_log", "min": 0.0001, "max": 0.1},
            {"name": "batch_size", "type": "categorical", "options": [4, 8, 16, 32]},
            {"name": "resolution", "type": "categorical", "options": [256, 512, 1024]},
            {"name": "loss_weight_ratio", "type": "float", "min": 0.0, "max": 1.0},
            {"name": "model_capacity", "type": "categorical", "options": ["narrow", "wide"]},
        ],
        "worker": {"entrypoint": "python simulators/training_worker.py"},
    }

    init_study_from_manifest_dict(DEMO_MANIFEST, force=True)

    broker_url = settings.broker_url or str(request.base_url).rstrip("/")
    Thread(
        target=run_training_worker,
        args=(DEMO_STUDY,),
        kwargs={"max_trials": 5, "broker_url": broker_url},
        daemon=True,
    ).start()

    return {"success": True, "study_name": DEMO_STUDY}


@router.get("/worker_snippet")
def api_worker_snippet(study_name: str):
    broker_url = settings.broker_url or "http://localhost:8000"
    secret_token = settings.secret_token
    return {
        "success": True,
        "broker_url": broker_url,
        "study_name": study_name,
        "auth_required": bool(secret_token),
        "snippet": (
            f"export HPO_BROKER_URL={broker_url}\n"
            f"export HPO_STUDY_NAME={study_name}\n"
            + (f"export HPO_SECRET_TOKEN={secret_token}\n" if secret_token else "")
            + "python your_worker.py"
        ),
    }
