import os
import json
import time
import hmac
import traceback
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import optuna
from optuna.trial import TrialState

from src.db_manager import get_db_session, DATABASE_URL, init_db
from src.schema import TrialResult, StudyStatus, SystemConfiguration, CoordinatorMetric, SuggestMetric
from src.hpo_config import load_hpo_config, save_hpo_config, normalize_trial_params, param_display_name
from src.metrics import score_objective_index, _trial_metric_snapshot
from src.hpo_coordinator import (
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
from src.search_space import (
    _migrate_search_space,
    load_search_space,
    _apply_search_space_patch,
    handle_api_get_search_space,
    handle_api_update_search_space,
)
from src.pruning import _effective_train_resolution
from src.suggest import (
    SuggestRequest,
    get_or_create_study,
    load_study,
    _enqueue_manual_trial,
    handle_api_suggest_trial,
)
from src.leases import (
    _reap_stale_running_trials,
    _reap_expired_leases,
    HeartbeatRequest,
    handle_api_heartbeat,
)
from src.reporting import (
    ReportEpochRequest,
    CompleteTrialRequest,
    handle_api_report_epoch,
    handle_api_complete_trial,
)

# Ensure custom tables (incl. study_reviews) exist before serving.
init_db()

app = FastAPI(title="Pathfinder HTTP Broker")
_js_dir = os.path.join(os.path.dirname(__file__), "web", "js")
app.mount("/js", StaticFiles(directory=_js_dir), name="js")

# --- CORS ---
def _allowed_origins() -> List[str]:
    """Explicit CORS allowlist: localhost defaults plus comma-separated HPO_ALLOWED_ORIGINS."""
    origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    extra = os.environ.get("HPO_ALLOWED_ORIGINS", "")
    origins += [o.strip() for o in extra.split(",") if o.strip()]
    return origins


secret_token_env = os.environ.get("HPO_SECRET_TOKEN")
if secret_token_env:
    # Token mode (e.g. tunneled): lock CORS to an explicit allowlist; allow credentials
    # (the session cookie) only for those origins. "*" + credentials is invalid per spec.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Local dev, no token configured: permissive, but no credentialed cross-origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Paths reachable without a token even when one is configured.
_AUTH_EXEMPT_PATHS = {"/", "/health", "/api/login", "/favicon.ico"}
# Non-/api routes that still serve potentially sensitive content and must be protected.
_AUTH_PROTECTED_EXACT = {"/colab_worker.py", "/hpo_client.py", "/worker_minimal.py"}


def _request_token(request: Request) -> Optional[str]:
    """Extract a caller-supplied token from header (workers/CLI) or session cookie (browser)."""
    token_header = request.headers.get("X-HPO-Token")
    if token_header:
        return token_header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):]
    return request.cookies.get("hpo_session")


@app.middleware("http")
async def hpo_secret_token_middleware(request: Request, call_next):
    secret_token = os.environ.get("HPO_SECRET_TOKEN")
    if secret_token:
        path = request.url.path
        protected = path not in _AUTH_EXEMPT_PATHS and (
            path.startswith("/api/") or path in _AUTH_PROTECTED_EXACT
        )
        if protected:
            provided = _request_token(request)
            if not provided or not hmac.compare_digest(provided, secret_token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "Unauthorized: provide a valid token via the X-HPO-Token header, or log in at /api/login to set a session cookie.",
                    },
                )
    response = await call_next(request)
    return response


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

# --- GUI ROOT ROUTE ---
@app.get("/", response_class=HTMLResponse)
def get_gui():
    gui_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(gui_path):
        with open(gui_path, "r") as f:
            html = f.read()
        # Inject only a NON-SECRET hint so the dashboard knows whether to prompt for login.
        # The token itself is never placed in the page; the browser authenticates via an
        # httpOnly session cookie set by POST /api/login.
        auth_required = "true" if os.environ.get("HPO_SECRET_TOKEN") else "false"
        inject_script = f"<script>window.HPO_AUTH_REQUIRED = {auth_required};</script>"
        if "</head>" in html:
            html = html.replace("</head>", f"{inject_script}</head>", 1)
        else:
            html = inject_script + html
        return html
    return "<h3>index.html not found</h3>"


@app.get("/styles.css")
def get_styles():
    styles_path = os.path.join(os.path.dirname(__file__), "web", "styles.css")
    if os.path.exists(styles_path):
        return FileResponse(styles_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/colab_worker.py")
def get_colab_worker():
    worker_path = os.path.join(os.path.dirname(__file__), "colab_worker.py")
    if os.path.exists(worker_path):
        return FileResponse(worker_path, media_type="text/x-python", filename="colab_worker.py")
    raise HTTPException(status_code=404, detail="colab_worker.py not found")

@app.get("/hpo_client.py")
def get_hpo_client():
    client_path = os.path.join(os.path.dirname(__file__), "src", "hpo_client.py")
    if os.path.exists(client_path):
        return FileResponse(client_path, media_type="text/x-python", filename="hpo_client.py")
    raise HTTPException(status_code=404, detail="hpo_client.py not found")

@app.get("/worker_minimal.py")
def get_worker_minimal():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "worker_minimal.py")
    if os.path.exists(template_path):
        return FileResponse(template_path, media_type="text/x-python", filename="worker_minimal.py")
    raise HTTPException(status_code=404, detail="worker_minimal.py not found")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "connected" if DATABASE_URL else "disconnected",
        "worker_endpoints": {
            "suggest_trial": "POST /api/suggest_trial",
            "report_epoch": "POST /api/report_epoch",
            "complete_trial": "POST /api/complete_trial",
        },
    }


class LoginRequest(BaseModel):
    token: str


@app.post("/api/login")
def api_login(req: LoginRequest, request: Request):
    """Exchange the shared token for an httpOnly session cookie (dashboard login).

    The token is never embedded in the page; the browser holds only an httpOnly cookie, so a
    dashboard XSS cannot exfiltrate it. Workers/CLI keep using the X-HPO-Token header.
    """
    secret_token = os.environ.get("HPO_SECRET_TOKEN")
    if not secret_token:
        # No auth configured (local dev): nothing to log in to.
        return JSONResponse({"success": True, "auth_required": False})
    if not hmac.compare_digest(req.token, secret_token):
        raise HTTPException(status_code=401, detail="Invalid token.")
    # Secure cookie only over https (tunnel); plain localhost http must still receive it.
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


@app.get("/api/suggest_trial")
def api_suggest_trial_help():
    """Browser/curl GET lands here — suggest requires POST (Colab worker uses POST)."""
    return {
        "error": "Method not allowed: use POST, not GET",
        "post_url": "/api/suggest_trial",
        "body_example": {
            "study_name": "bridge_crack_study",
            "reasoning": "Autonomous worker suggestion request.",
        },
        "curl_example": (
            'curl -X POST "$BROKER_URL/api/suggest_trial" '
            '-H "Content-Type: application/json" '
            '-d \'{"study_name":"bridge_crack_study"}\''
        ),
    }

# --- ACTIVE CONFIGURATION & SEARCH SPACE CONTROLS ENDPOINTS ---
@app.get("/api/hpo_config")
def api_get_hpo_config(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    return load_hpo_config(study_name)

@app.post("/api/hpo_config")
def api_save_hpo_config(config: Dict[str, Any], study_name: Optional[str] = None):
    if not study_name:
        study_name = config.get("study_name") or os.getenv("HPO_STUDY_NAME", "seg_v1")
    config_clean = {k: v for k, v in config.items() if k != "study_name"}
    save_hpo_config(config_clean, study_name)
    return {"success": True, "config": load_hpo_config(study_name)}


@app.get("/api/search_space")
def api_get_search_space(study_name: Optional[str] = None):
    return handle_api_get_search_space(study_name)

@app.post("/api/update_search_space")
def api_update_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    return handle_api_update_search_space(space, study_name)


@app.get("/api/study_health")
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
            except Exception:
                pass
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


@app.get("/api/pending_changes")
def api_get_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
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


@app.post("/api/apply_pending_changes")
def api_apply_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
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


@app.post("/api/discard_pending_changes")
def api_discard_pending_changes(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
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


class InitFromManifestRequest(BaseModel):
    yaml: str

@app.post("/api/validate_manifest")
def api_validate_manifest(req: InitFromManifestRequest):
    import yaml
    from src.manifest import validate_manifest
    try:
        data = yaml.safe_load(req.yaml)
    except Exception as e:
        return {"success": False, "errors": [f"Invalid YAML structure: {str(e)}"], "warnings": []}
        
    if not isinstance(data, dict):
        return {"success": False, "errors": ["Manifest root must be a dictionary"], "warnings": []}

    errors, warnings = validate_manifest(data)
    return {"success": len(errors) == 0, "errors": errors, "warnings": warnings}

@app.post("/api/init_from_manifest")
def api_init_from_manifest(req: InitFromManifestRequest, force: bool = False):
    import yaml
    from src.manifest import validate_manifest
    from src.onboarding import init_study_from_manifest_dict
    
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


@app.get("/api/studies")
def api_list_studies():
    try:
        summaries = optuna.get_all_study_summaries(storage=DATABASE_URL)
        return {"success": True, "studies": [s.study_name for s in summaries]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/study_setup")
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
            "manifest_metrics": hpo_config.get("manifest_metrics")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- METRICS & VISUALIZATION ENDPOINTS ---
@app.get("/api/study_details")
def api_study_details(study_name: str):
    try:
        study = get_or_create_study(study_name)

        with get_db_session() as session:
            _reap_expired_leases(study, study_name, session)

        with get_db_session() as session:
            metric_rows = session.query(TrialResult).filter_by(study_name=study_name).all()
            metrics_dict = {m.trial_id: m.get_history() for m in metric_rows}
            # Build a lookup for OOM / failure fields by Optuna trial_id using plain dictionaries
            # to avoid DetachedInstanceError outside the session block.
            trial_result_map = {
                m.trial_id: {
                    "oom_triggered": m.oom_triggered,
                    "failure_tag": m.failure_tag,
                    "gpu_model": m.gpu_model,
                    "max_vram_gb": m.max_vram_gb,
                }
                for m in metric_rows
            }

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

            # OOM / failure fields from TrialResult
            tr = trial_result_map.get(t._trial_id)
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
                "oom_triggered": (tr.get("oom_triggered") if tr else None) or t.user_attrs.get("oom_triggered", False),
                "failure_tag": tr.get("failure_tag") if tr else None,
                "gpu_model": tr.get("gpu_model") if tr else None,
                "max_vram_gb": tr.get("max_vram_gb") if tr else None,
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

@app.get("/api/fanova")
def api_fanova(study_name: str):
    try:
        study = get_or_create_study(study_name)
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(complete_trials) < 2:
            return {"success": False, "message": "Need at least 2 completed trials to compute fANOVA."}
            
        importances = {}
        if len(study.directions) > 1:
            score_idx = score_objective_index(study)
            if score_idx is not None:
                importances = optuna.importance.get_param_importances(
                    study,
                    target=lambda t, _idx=score_idx: t.values[_idx] if (t.values and len(t.values) > _idx) else None,
                    evaluator=optuna.importance.FanovaImportanceEvaluator()
                )
        else:
            importances = optuna.importance.get_param_importances(
                study,
                evaluator=optuna.importance.FanovaImportanceEvaluator()
            )
            
        config = load_hpo_config()
        aliases = config.get("legacy_param_aliases", {})
        display: Dict[str, float] = {}
        for param, value in importances.items():
            canonical = aliases.get(param, param)
            label = param_display_name(canonical, config)
            if label in display:
                display[label] = max(display[label], value)
            else:
                display[label] = value

        return {"success": True, "importances": display}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/api/review_packet")
def api_review_packet(study_name: str):
    """Read-only context for the IDE coordinator: Pareto, fANOVA, eval insights, drift reasons."""
    try:
        return build_review_packet(study_name)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pareto_front")
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

@app.post("/api/dismiss_coordinator_nudge")
def api_dismiss_coordinator_nudge(study_name: str):
    """Dismisses the coordinator nudge for the current trial window by persisting it in SQLite."""
    try:
        from src.db_manager import get_db_session
        from src.schema import StudyStatus
        from src.hpo_coordinator import count_evaluated_trials
        
        study = get_or_create_study(study_name)
        trials_evaluated = count_evaluated_trials(study)
        
        with get_db_session() as session:
            status = session.query(StudyStatus).filter_by(study_name=study_name).first()
            if not status:
                status = StudyStatus(study_name=study_name)
                session.add(status)
            status.nudge_dismissed_trials = trials_evaluated
            session.commit()
            
        return {"success": True, "dismissed_trials": trials_evaluated}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/study_cards")
def api_get_study_cards(study_name: Optional[str] = None):
    """Exposes generated study cards and their markdown content for dashboard retrieval."""
    try:
        from src.hpo_coordinator import load_study_cards
        return {"success": True, "cards": load_study_cards(study_name)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent_review")
def api_agent_review(req: AgentReviewRequest):
    """Persist a coordinator review. Idempotent per trial window unless force=True.

    Optionally applies a search-space patch and/or enqueues a single manual trial as the
    chosen policy action. Never blocks the worker suggest path.
    """
    try:
        study = load_study(req.study_name)
        space = load_search_space(req.study_name)
        trials_evaluated = count_evaluated_trials(study)

        if req.manual_trial:
            from src.hpo_coordinator import _validate_manual_parameters
            val_res = _validate_manual_parameters(req.manual_trial, req.study_name)
            if not val_res["ok"]:
                with get_db_session() as session:
                    from src.schema import InvalidProposal
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


@app.post("/api/flag_review")
def api_flag_review(review_id: int, flagged: bool = True):
    """Mark a coordinator review as low-quality (excluded from accuracy MAE)."""
    try:
        return flag_study_review(review_id, flagged=flagged)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- WORKER HPO ROUTER ENDPOINTS ---
@app.post("/api/suggest_trial")
@app.post("/api/suggest_trials")  # common typo alias
def api_suggest_trial(req: SuggestRequest):
    return handle_api_suggest_trial(req)


@app.post("/api/heartbeat")
def api_heartbeat(req: HeartbeatRequest):
    return handle_api_heartbeat(req)


@app.post("/api/report_epoch")
def api_report_epoch(req: ReportEpochRequest):
    return handle_api_report_epoch(req)

@app.post("/api/complete_trial")
def api_complete_trial(req: CompleteTrialRequest):
    return handle_api_complete_trial(req)

@app.get("/api/tunnel_url")
def api_get_tunnel_url():
    """Returns the active remote broker URL if established (Cloudflare, ngrok, or Tailscale)."""
    try:
        with get_db_session() as session:
            # Try key remote_broker_url first, fallback to ngrok_tunnel_url
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

@app.get("/api/debug/config_audit")
def api_config_audit(study_name: Optional[str] = None):
    # Debug-only endpoint; off unless explicitly enabled to avoid leaking internal config state.
    if os.getenv("HPO_DEBUG") != "1":
        raise HTTPException(status_code=404, detail="Not found.")
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    
    report = {
        "study_name": study_name,
        "violations": [],
    }

    db_space = load_search_space(study_name)

    # Trial parameter validation: flag any trial whose params fall outside the active space.
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

@app.get("/api/metrics/coordinator")
def api_metrics_coordinator(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    with get_db_session() as session:
        rows = session.query(CoordinatorMetric).filter_by(study_name=study_name).all()
        return {"success": True, "metrics": [r.to_dict() for r in rows]}

@app.get("/api/metrics/suggest")
def api_metrics_suggest(study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    with get_db_session() as session:
        rows = session.query(SuggestMetric).filter_by(study_name=study_name).all()
        return {"success": True, "metrics": [r.to_dict() for r in rows]}

@app.get("/api/mcp_info")
def api_mcp_info():
    return {
        "success": True,
        "mcp_server_name": "pathfinder",
        "active_study": os.getenv("HPO_STUDY_NAME", "seg_v1"),
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

# Backward-compat re-exports — prefer importing from src.* in new code.
from src.search_space import save_search_space  # noqa: F401
from src.leases import _try_claim_lease  # noqa: F401
from src.pruning import _epoch_composite_score, _pruning_peer_trials  # noqa: F401

if __name__ == "__main__":
    import argparse
    import uvicorn
    import threading
    import os
    
    parser = argparse.ArgumentParser(description="Pathfinder HTTP Broker")
    parser.add_argument("--host", default="127.0.0.1", help="Binding host (default loopback-only)")
    parser.add_argument("--port", type=int, default=8000, help="Binding port")
    parser.add_argument("--daemon", action="store_true", help="Start background health daemon thread")
    parser.add_argument("--tunnel", action="store_true", help="Start background ngrok tunnel")
    parser.add_argument("--tunnel-provider", default=None, choices=["ngrok", "cloudflare", "none"], help="Tunneling provider (ngrok, cloudflare, or none)")
    parser.add_argument("--tunnel-url", default=None, help="Pre-configured static remote tunnel/broker URL (e.g. Cloudflare custom domain)")
    
    args = parser.parse_args()

    # Resolve tunnel provider
    tunnel_provider = args.tunnel_provider
    if tunnel_provider is None:
        env_provider = os.getenv("HPO_TUNNEL_PROVIDER")
        if env_provider:
            tunnel_provider = env_provider.lower()
        elif args.tunnel or os.getenv("HPO_TUNNEL_ENABLED") == "1":
            tunnel_provider = "ngrok"
        else:
            tunnel_provider = "none"

    # Resolve static tunnel URL
    static_tunnel_url = args.tunnel_url or os.getenv("HPO_TUNNEL_URL")
    if static_tunnel_url and tunnel_provider == "none":
        tunnel_provider = "cloudflare"

    # Network safety: refuse to expose the broker beyond loopback (or via a tunnel) without a
    # token. This is the single most important guard for the tunneled threat model.
    tunnel_requested = (tunnel_provider != "none")
    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    
    if (not is_loopback or tunnel_requested) and not os.environ.get("HPO_SECRET_TOKEN"):
        raise SystemExit(
            "Refusing to start: exposing the broker beyond loopback (non-loopback --host for Tailscale or "
            "tunneling for Cloudflare/ngrok) without authentication is unsafe.\n"
            "Please set the HPO_SECRET_TOKEN environment variable to secure the broker, "
            "or bind to local loopback only (use --host 127.0.0.1) for local-only use."
        )
    
    if not is_loopback and os.environ.get("HPO_SECRET_TOKEN"):
        print(f"🔒 Secure Private VPN/Tailscale Network Mode enabled. Binding to {args.host}:{args.port}")
        print("   HPO_SECRET_TOKEN is active. Workers must supply a valid token to connect.")

    # 1. Start daemon thread if requested (notify-only health monitor; no auto-LLM).
    daemon_enabled = args.daemon or os.getenv("HPO_DAEMON_ENABLED") == "1"

    if daemon_enabled:
        from src.hpo_daemon import run_daemon_loop
        d_thread = threading.Thread(
            target=run_daemon_loop,
            kwargs={"interval_seconds": 10},
            daemon=True
        )
        d_thread.start()
        
    # 2. Process Static Tunnel / Cloudflare mode
    if tunnel_provider == "cloudflare" or static_tunnel_url:
        if not static_tunnel_url:
            print("Error: --tunnel-url or HPO_TUNNEL_URL environment variable must be provided for Cloudflare/static tunnel provider.")
            raise SystemExit(1)
        
        # Persist the static URL in SQLite under SystemConfiguration so dashboard and workers can fetch it
        try:
            from src.db_manager import get_db_session
            from src.schema import SystemConfiguration
            with get_db_session() as session:
                session.merge(SystemConfiguration(
                    study_name="_global",
                    config_key="remote_broker_url",
                    config_value=static_tunnel_url
                ))
                # For backwards compatibility, write to the old key too
                session.merge(SystemConfiguration(
                    study_name="_global",
                    config_key="ngrok_tunnel_url",
                    config_value=static_tunnel_url
                ))
                session.commit()
            print(f"\n==============================================")
            print(f"🔥 Remote broker URL established (Static/Cloudflare): {static_tunnel_url}")
            print(f"==============================================\n")
        except Exception as db_err:
            print(f"Error saving remote broker URL: {db_err}")

    # 3. Start ngrok tunnel if requested and no static URL was provided
    elif tunnel_provider == "ngrok":
        import subprocess
        import requests
        import time
        
        def start_ngrok(port):
            try:
                print(f"Spawning ngrok tunnel for port {port}...")
                proc = subprocess.Popen(
                    ["ngrok", "http", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2.0)
                # Query local agent API
                for _ in range(10):
                    try:
                        res = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=2)
                        if res.status_code == 200:
                            tunnels = res.json().get("tunnels", [])
                            for t in tunnels:
                                if t.get("proto") == "https":
                                    public_url = t.get("public_url")
                                    print(f"\n==============================================")
                                    print(f"🔥 Ngrok tunnel established: {public_url}")
                                    print(f"==============================================\n")
                                    
                                    # Persist in SQLite under SystemConfiguration
                                    try:
                                        from src.db_manager import get_db_session
                                        from src.schema import SystemConfiguration
                                        with get_db_session() as session:
                                            session.merge(SystemConfiguration(
                                                study_name="_global",
                                                config_key="remote_broker_url",
                                                config_value=public_url
                                            ))
                                            session.merge(SystemConfiguration(
                                                study_name="_global",
                                                config_key="ngrok_tunnel_url",
                                                config_value=public_url
                                            ))
                                            session.commit()
                                    except Exception as db_err:
                                        print(f"Error saving tunnel url: {db_err}")
                                    return
                    except Exception:
                        time.sleep(1.0)
                print("Warning: Ngrok started but local agent API did not report tunnel URL.")
            except FileNotFoundError:
                print("Warning: 'ngrok' command not found in PATH. Please install it or run ngrok manually.")
            except Exception as e:
                print(f"Failed to start ngrok: {e}")
                
        t_thread = threading.Thread(target=start_ngrok, args=(args.port,), daemon=True)
        t_thread.start()

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)
