import os
import json
import time
import hmac
import traceback
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState
from sqlalchemy import create_engine, text, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db_manager import get_db_session, DATABASE_URL, init_db
from src.schema import TrialResult, AgentReasoningLog, StudyStatus, SystemConfiguration, TrialLease, CoordinatorMetric, SuggestMetric
from src.hpo_config import load_hpo_config, save_hpo_config, normalize_trial_params, param_display_name
from src.hpo_coordinator import (
    trial_train_resolution as _trial_train_resolution,
    study_eval_insights as _study_eval_insights,
    pareto_trial_numbers_deploy_aware as _pareto_trial_numbers_deploy_aware,
    compute_health_tier,
    build_review_packet,
    save_study_review,
    get_latest_study_review,
    get_recent_study_reviews,
    count_evaluated_trials,
    build_review_prompt,
    POLICY_ACTIONS,
    compute_review_heuristics,
    compute_statistical_confidence,
    validate_review_fields,
    backfill_review_outcomes,
    mark_review_applied,
    flag_study_review,
)

# Ensure custom tables (incl. study_reviews) exist before serving.
init_db()

app = FastAPI(title="Pathfinder HTTP Broker")

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

# Search space lives in SQLite (system_configuration); there is no on-disk config file.
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float_log"},
    "batch_size": {"options": [2, 4, 8, 16, 32, 64], "active": [2, 4, 8, 16, 32, 64], "type": "categorical"},
    "resolution": {"options": [256, 512, 1024], "active": [256, 512, 1024], "type": "categorical"},
    "model_capacity": {"options": ["narrow", "wide"], "active": ["narrow", "wide"], "type": "categorical"},
    "loss_weight_ratio": {"min": 0.0, "max": 1.0, "type": "float"},
}


def _migrate_search_space(space: Dict[str, Any], study_name: Optional[str] = None) -> Dict[str, Any]:
    """Rename legacy encoder_name → model_capacity for older studies/files."""
    config = load_hpo_config(study_name)
    if "encoder_name" in space and "model_capacity" not in space:
        enc = space.pop("encoder_name")
        mapping = config.get("legacy_capacity_values", {})
        enc["active"] = [mapping.get(v, v) for v in enc.get("active", enc.get("options", []))]
        enc["options"] = [mapping.get(v, v) for v in enc.get("options", enc.get("active", []))]
        space["model_capacity"] = enc
    return space


def load_search_space(study_name: Optional[str] = None) -> Dict[str, Any]:
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    
    try:
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="active_search_space"
            ).first()
            if row:
                space = json.loads(row.config_value)
                return _migrate_search_space(space, study_name)
    except Exception as e:
        print(f"Error loading search space from DB: {e}")

    # Seed it in DB if not found
    try:
        save_search_space(DEFAULT_SEARCH_SPACE, study_name)
    except Exception:
        pass
    return DEFAULT_SEARCH_SPACE.copy()


def _expected_search_params(space: Dict[str, Any]) -> List[str]:
    return [k for k, v in space.items() if isinstance(v, dict) and v.get("type")]


def _trial_has_full_params(trial_or_params, space: Dict[str, Any]) -> bool:
    """Accept a FrozenTrial or a finalized params dict."""
    params = (
        trial_or_params
        if isinstance(trial_or_params, dict)
        else trial_or_params.params
    )
    expected = set(_expected_search_params(space))
    return expected.issubset(params.keys())


def _worker_ready_params(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    return _finalize_trial_params(dict(trial.params), space)


def _cleanup_stuck_running_trials(study, space: Dict[str, Any]) -> None:
    """Fail RUNNING trials that never received a full parameter set (crashed mid-suggest)."""
    for t in list(study.trials):
        if t.state == TrialState.RUNNING and not _trial_has_full_params(
            _worker_ready_params(t, space), space
        ):
            print(
                f"Failing stuck RUNNING trial #{t.number} (incomplete params: {list(t.params.keys())})"
            )
            try:
                study.tell(t.number, state=TrialState.FAIL)
            except Exception as exc:
                print(f"Could not fail trial #{t.number}: {exc}")


def _study_categorical_choices(study, param: str, cfg: Dict[str, Any]) -> List[Any]:
    """
    Optuna forbids narrowing categorical distributions after the first trial.
    Always merge JSON options with choices already committed in this study.
    """
    options = list(cfg.get("options") or cfg.get("active") or [])
    historical: List[Any] = []
    for t in study.trials:
        dist = t.distributions.get(param)
        if dist is not None and hasattr(dist, "choices"):
            historical = list(dist.choices)
            break
    merged = list(dict.fromkeys(historical + options))
    return merged if merged else options


def _fixed_categorical_params(space: Dict[str, Any]) -> Dict[str, Any]:
    """Params with a single active categorical value (not passed through suggest_categorical)."""
    fixed: Dict[str, Any] = {}
    for param, cfg in space.items():
        if not isinstance(cfg, dict) or cfg.get("type") != "categorical":
            continue
        active = list(cfg.get("active") or cfg.get("options") or [])
        if len(active) == 1:
            fixed[param] = active[0]
    return fixed


def _finalize_trial_params(params: Dict[str, Any], space: Dict[str, Any]) -> Dict[str, Any]:
    """Merge fixed single-choice categoricals into the param dict returned to workers."""
    out = dict(params)
    out.update(_fixed_categorical_params(space))
    return out


def _persist_fixed_categorical_params(study, trial, space: Dict[str, Any]) -> None:
    """Write single-active categoricals into Optuna storage so dashboards see them."""
    fixed = _fixed_categorical_params(space)
    for param, value in fixed.items():
        if param in trial.params:
            continue
        cfg = space.get(param, {})
        choices = tuple(_study_categorical_choices(study, param, cfg))
        dist = CategoricalDistribution(choices=choices)
        # Optuna RDB storage expects internal index (0..n-1), not external choice value.
        internal = float(dist.to_internal_repr(value))
        study._storage.set_trial_param(trial._trial_id, param, internal, dist)
        study._storage.set_trial_user_attr(trial._trial_id, param, value)


def _effective_train_resolution(
    trial, hpo_config: Dict[str, Any], space: Dict[str, Any]
) -> Optional[Any]:
    train_param = hpo_config.get("eval_protocol", {}).get("train_resolution_param", "resolution")
    val = trial.params.get(train_param)
    if val is not None:
        return val
    val = trial.user_attrs.get(train_param)
    if val is not None:
        return val
    res_cfg = space.get(train_param) or space.get("resolution")
    if isinstance(res_cfg, dict) and res_cfg.get("type") == "categorical":
        active = res_cfg.get("active") or []
        if len(active) == 1:
            return active[0]
    return None


def _trial_metric_snapshot(
    trial,
    history: List[Dict[str, Any]],
    dice_fixed_attr: str,
    bce_fixed_attr: str,
    directions: List[Any] = None,
) -> Dict[str, Any]:
    """Dice/BCE for dashboard: completed values, else latest epoch / user_attrs."""
    bce = dice = dice_eval_fixed = bce_eval_fixed = None
    latest_epoch = trial.user_attrs.get("latest_epoch")

    from optuna.study import StudyDirection

    if trial.state == TrialState.COMPLETE and (trial.values or trial.value is not None):
        if trial.values and len(trial.values) > 1:
            bce, dice = trial.values[0], trial.values[1]
        else:
            if directions and directions[0] == StudyDirection.MINIMIZE:
                bce = trial.value
            else:
                dice = trial.value
    else:
        dice = trial.user_attrs.get("latest_dice")
        bce = trial.user_attrs.get("latest_bce")
        dice_eval_fixed = trial.user_attrs.get(dice_fixed_attr)
        bce_eval_fixed = trial.user_attrs.get(bce_fixed_attr)

    if history:
        last = max(history, key=lambda e: e.get("epoch", 0))
        latest_epoch = latest_epoch or last.get("epoch")
        if dice is None:
            dice = last.get("dice")
        if bce is None:
            bce = last.get("bce")
        if dice_eval_fixed is None:
            dice_eval_fixed = last.get("dice_eval_fixed")
        if bce_eval_fixed is None:
            bce_eval_fixed = last.get("bce_eval_fixed")

    if dice_eval_fixed is None:
        dice_eval_fixed = trial.user_attrs.get(dice_fixed_attr)
    if bce_eval_fixed is None:
        bce_eval_fixed = trial.user_attrs.get(bce_fixed_attr)

    return {
        "bce": bce,
        "dice": dice,
        "dice_eval_fixed": dice_eval_fixed,
        "bce_eval_fixed": bce_eval_fixed,
        "latest_epoch": latest_epoch,
    }


def _enqueue_single_active_categoricals(study, space: Dict[str, Any]) -> None:
    """Optional hint for Optuna; workers still receive fixed values via _finalize_trial_params."""
    fixed = _fixed_categorical_params(space)
    if fixed:
        study.enqueue_trial(fixed)


def _suggest_categorical_compatible(study, trial, param: str, cfg: Dict[str, Any]) -> None:
    active = list(cfg.get("active") or cfg.get("options") or [])
    if not active:
        raise ValueError(f"Categorical parameter '{param}' has no active options.")
    if len(active) == 1:
        # Set via enqueue_trial before study.ask(); do not call suggest here.
        return
    choices = _study_categorical_choices(study, param, cfg)
    trial.suggest_categorical(param, choices)


def _validate_params_against_active(params: Dict[str, Any], space: Dict[str, Any]) -> List[str]:
    """Return list of human-readable violations when TPE samples outside active constraints."""
    errors = []
    for param, cfg in space.items():
        if not isinstance(cfg, dict) or cfg.get("type") != "categorical":
            continue
        active = set(cfg.get("active") or cfg.get("options") or [])
        if param in params and params[param] not in active:
            errors.append(
                f"{param}={params[param]!r} not in active {sorted(active)} "
                f"(Optuna still uses full historical choices for this study)"
            )
    return errors


def suggest_params_from_space(study, trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest all parameters defined in the active search space JSON."""
    for param, cfg in space.items():
        if not isinstance(cfg, dict) or "type" not in cfg:
            continue
        ptype = cfg["type"]
        if ptype == "float_log":
            trial.suggest_float(param, float(cfg["min"]), float(cfg["max"]), log=True)
        elif ptype == "float":
            trial.suggest_float(param, float(cfg["min"]), float(cfg["max"]))
        elif ptype == "categorical":
            _suggest_categorical_compatible(study, trial, param, cfg)
        else:
            raise ValueError(f"Unsupported parameter type '{ptype}' for '{param}'.")
    params = _finalize_trial_params(trial.params, space)
    violations = _validate_params_against_active(params, space)
    if violations:
        raise ValueError(
            "Sampled parameters outside active search bounds: "
            + "; ".join(violations)
            + ". Start a new study or widen active options."
        )
    return params

def save_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    try:
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="active_search_space"
            ).first()
            if row:
                row.config_value = json.dumps(space)
                row.version += 1
            else:
                session.add(SystemConfiguration(
                    study_name=study_name,
                    config_key="active_search_space",
                    config_value=json.dumps(space),
                    version=1
                ))
    except Exception as e:
        print(f"Error saving search space to DB: {e}")





def _epoch_composite_score(study, trial, epoch: int, ev: Dict[str, Any]) -> Optional[float]:
    """Composite score at epoch, Z-score normalized against study rolling history at the same epoch, fallback to (dice - bce)."""
    import numpy as np
    
    use_fixed = ev.get("enabled") and ev.get("use_fixed_metric_for_pruning")
    
    # 1. Retrieve current trial's metrics at this epoch
    curr_score = None
    curr_loss = None
    history = trial.user_attrs.get("history", [])
    if isinstance(history, list):
        for entry in history:
            if entry.get("epoch") == epoch:
                if use_fixed and entry.get("dice_eval_fixed") is not None:
                    curr_score = entry["dice_eval_fixed"]
                    curr_loss = entry.get("bce_eval_fixed", entry.get("bce", 0.0))
                else:
                    curr_score = entry.get("dice")
                    curr_loss = entry.get("bce")
                break
                
    if curr_score is None or curr_loss is None:
        if epoch in trial.intermediate_values:
            return float(trial.intermediate_values[epoch])
        return None

    # 2. Gather metrics at the SAME epoch across all Complete/Running trials
    scores = []
    losses = []
    for t in study.trials:
        if t.state not in (TrialState.COMPLETE, TrialState.RUNNING):
            continue
        t_history = t.user_attrs.get("history", [])
        if isinstance(t_history, list):
            for entry in t_history:
                if entry.get("epoch") == epoch:
                    if use_fixed and entry.get("dice_eval_fixed") is not None:
                        s = entry["dice_eval_fixed"]
                        l = entry.get("bce_eval_fixed", entry.get("bce", 0.0))
                    else:
                        s = entry.get("dice")
                        l = entry.get("bce")
                    if s is not None and l is not None:
                        scores.append(float(s))
                        losses.append(float(l))
                    break

    # 3. Z-score normalize if we have enough history (>= 10 values)
    if len(scores) < 10:
        # Not enough history: return score only (conservative fallback)
        return float(curr_score)

    score_mean, score_std = np.mean(scores), np.std(scores)
    loss_mean, loss_std = np.mean(losses), np.std(losses)

    # Tweak 2: Clamp standard deviation to prevent division by zero in early homogeneous trials
    score_std = score_std if score_std > 1e-6 else 1.0
    loss_std = loss_std if loss_std > 1e-6 else 1.0

    z_score = (float(curr_score) - score_mean) / score_std
    z_loss = -(float(curr_loss) - loss_mean) / loss_std

    return float(z_score + z_loss)


def _pruning_peer_trials(study, trial_obj, hpo_config: Dict[str, Any]) -> List:
    ev = hpo_config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    current_res = _trial_train_resolution(trial_obj, train_param)
    low_warn = ev.get("low_train_res_warning")
    low_warn = int(low_warn) if low_warn is not None else None
    same_only = ev.get("prune_compare_same_resolution_only", True)
    exclude_low = ev.get("prune_exclude_low_res_from_baseline", True)

    peers = []
    for t in study.trials:
        if t.number == trial_obj.number:
            continue
        if t.state not in (TrialState.COMPLETE, TrialState.RUNNING):
            continue
        peer_res = _trial_train_resolution(t, train_param)
        if same_only and current_res is not None and peer_res is not None:
            if peer_res != current_res:
                continue
        if (
            exclude_low
            and low_warn is not None
            and current_res is not None
            and current_res >= low_warn
            and peer_res is not None
            and peer_res < low_warn
        ):
            continue
        peers.append(t)
    return peers


class SuggestRequest(BaseModel):
    study_name: str
    worker_id: Optional[str] = None
    agent_model: Optional[str] = "optuna-tpe"
    prompt_strategy: Optional[str] = "tpe_sampler"
    reasoning: Optional[str] = "Autonomous worker suggestion request."
    estimated_score_improvement: Optional[float] = None
    estimated_dice_improvement: Optional[float] = 0.0

class ReportEpochRequest(BaseModel):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    score: Optional[float] = None
    loss: Optional[float] = None
    dice: Optional[float] = None
    bce: Optional[float] = None
    gpu_memory: Optional[float] = 0.0
    speed_ips: Optional[float] = 0.0
    score_eval_fixed: Optional[float] = None
    loss_eval_fixed: Optional[float] = None
    dice_eval_fixed: Optional[float] = None
    bce_eval_fixed: Optional[float] = None

class CompleteTrialRequest(BaseModel):
    study_name: str
    trial_id: int
    worker_id: Optional[str] = None
    epoch: int
    score: Optional[float] = None
    loss: Optional[float] = None
    dice: Optional[float] = None
    bce: Optional[float] = None
    weights_path: str
    history: List[Dict[str, Any]]
    state: Optional[str] = "COMPLETE"
    gpu_memory: Optional[float] = 0.0
    speed_ips: Optional[float] = 0.0
    score_eval_fixed: Optional[float] = None
    loss_eval_fixed: Optional[float] = None
    dice_eval_fixed: Optional[float] = None
    bce_eval_fixed: Optional[float] = None
    gpu_model: Optional[str] = None
    max_vram_gb: Optional[float] = None
    oom_triggered: Optional[bool] = None

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
    estimated_dice_improvement: Optional[float] = None
    cited_best_trial: Optional[int] = None
    force: Optional[bool] = False

def _repair_categorical_param_indices(session: Session, study_name: str) -> int:
    """
    Fix trial_params rows where a categorical was stored as the external value
    (e.g. resolution=512.0) instead of Optuna's internal choice index.
    """
    fixed = 0
    rows = session.execute(
        text(
            """
            SELECT tp.param_id, tp.param_name, tp.param_value, tp.distribution_json
            FROM trial_params tp
            JOIN trials t ON t.trial_id = tp.trial_id
            JOIN studies s ON s.study_id = t.study_id
            WHERE s.study_name = :study_name
              AND tp.distribution_json LIKE '%CategoricalDistribution%'
            """
        ),
        {"study_name": study_name},
    ).fetchall()
    for param_id, param_name, param_value, dist_json in rows:
        try:
            dist_data = json.loads(dist_json)
            choices = dist_data["attributes"]["choices"]
            idx = int(float(param_value))
            if 0 <= idx < len(choices):
                continue
            external = float(param_value)
            if external not in choices and int(external) not in choices:
                continue
            value = int(external) if int(external) in choices else external
            internal = choices.index(value)
            session.execute(
                text(
                    "UPDATE trial_params SET param_value = :v WHERE param_id = :id"
                ),
                {"v": str(float(internal)), "id": param_id},
            )
            fixed += 1
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    session.commit()
    return fixed


def get_or_create_study(study_name: str):
    try:
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    except KeyError:
        print(f"Study '{study_name}' not found. Initializing new multi-objective study...")
        return optuna.create_study(
            study_name=study_name,
            storage=DATABASE_URL,
            directions=["minimize", "maximize"],
            load_if_exists=True
        )


def load_study(study_name: str):
    """Load an existing Optuna study, or raise 404 if it was never initialized.

    Worker and write endpoints use this (instead of get_or_create_study) so a typo'd or
    uninitialized study name fails loudly with a 404 rather than silently spawning an empty
    study. Studies are created up-front via the MCP initialize_study tool.
    """
    try:
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Study '{study_name}' is not initialized. Create it first via the MCP initialize_study tool.",
        )


# ~3× worker heartbeat interval (15s in hpo_client). Clears ghost RUNNING rows soon after crash.
LEASE_TTL_SECONDS = 45


def _reap_stale_running_trials(study, study_name: str, session) -> int:
    """Fail RUNNING trials whose worker lease expired or has no active lease."""
    from datetime import datetime

    now = datetime.utcnow()
    expired_trial_ids = [
        row.trial_id
        for row in session.query(TrialLease.trial_id).filter(
            TrialLease.study_name == study_name,
            TrialLease.lease_expires_at < now,
        ).all()
    ]
    active_leased_ids = {
        row.trial_id
        for row in session.query(TrialLease.trial_id).filter(
            TrialLease.study_name == study_name,
            TrialLease.lease_expires_at >= now,
        ).all()
    }
    reaped = 0
    for t in study.trials:
        if t.state != TrialState.RUNNING:
            continue
        stale = t._trial_id in expired_trial_ids or t._trial_id not in active_leased_ids
        if not stale:
            continue
        try:
            study.tell(t.number, state=TrialState.FAIL)
            reaped += 1
            reason = "expired lease" if t._trial_id in expired_trial_ids else "no active lease"
            print(f"Reaped stale RUNNING trial #{t.number} in '{study_name}' ({reason}).")
        except Exception as fail_err:
            print(f"Could not fail stale trial #{t.number}: {fail_err}")
    if expired_trial_ids:
        session.query(TrialLease).filter(
            TrialLease.trial_id.in_(expired_trial_ids)
        ).delete(synchronize_session=False)
    return reaped


# Backwards-compatible alias for callers/tests
_reap_expired_leases = _reap_stale_running_trials


def _try_claim_lease(session, study_name: str, trial_id: int, worker_id: str,
                     ttl_seconds: int = LEASE_TTL_SECONDS) -> bool:
    """Atomically claim/refresh a trial's lease for ``worker_id``. Returns True iff we won.

    A single conditional UPDATE succeeds only when the lease is currently free, expired, or
    already held by this worker. If no row matched (no lease yet), INSERT and win on the
    trial_id primary key; a losing racer hits IntegrityError and returns False. This is the
    sole mechanism preventing two workers from being handed the same trial. Safe under
    concurrency: SQLite (WAL + busy_timeout) serializes writers and Postgres locks the row.
    """
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    new_expiry = now + timedelta(seconds=ttl_seconds)
    updated = session.query(TrialLease).filter(
        TrialLease.trial_id == trial_id,
        or_(
            TrialLease.leased_to.is_(None),
            TrialLease.lease_expires_at < now,
            TrialLease.leased_to == worker_id,
        ),
    ).update(
        {"leased_to": worker_id, "lease_expires_at": new_expiry, "study_name": study_name},
        synchronize_session=False,
    )
    if updated == 1:
        return True
    try:
        session.add(TrialLease(
            trial_id=trial_id,
            study_name=study_name,
            leased_to=worker_id,
            lease_expires_at=new_expiry,
        ))
        session.flush()
        return True
    except IntegrityError:
        session.rollback()
        return False

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
    return load_search_space(study_name)

@app.post("/api/update_search_space")
def api_update_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    if not study_name:
        study_name = space.get("study_name") or os.getenv("HPO_STUDY_NAME", "seg_v1")
    space_clean = {k: v for k, v in space.items() if k != "study_name"}
    
    current = load_search_space(study_name)
    validated_proposals = {}
    
    for param_name, new_val in space_clean.items():
        if param_name in current:
            param_type = current[param_name].get("type")
            if param_type == "categorical":
                if "active" in new_val:
                    allowed = current[param_name].get("options", [])
                    invalid_options = [x for x in new_val["active"] if x not in allowed]
                    if invalid_options:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Active choices {invalid_options} for {param_name} are not in options: {allowed}"
                        )
                    if len(new_val["active"]) == 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Categorical parameter {param_name} must have at least one active option."
                        )
                    # Only add to proposal if it actually changed
                    if set(new_val["active"]) != set(current[param_name].get("active", [])):
                        validated_proposals[param_name] = {"active": new_val["active"]}
            else:
                proposal_param = {}
                if "min" in new_val:
                    val_min = float(new_val["min"])
                    if val_min != float(current[param_name].get("min", val_min)):
                        proposal_param["min"] = val_min
                if "max" in new_val:
                    val_max = float(new_val["max"])
                    if val_max != float(current[param_name].get("max", val_max)):
                        proposal_param["max"] = val_max
                if proposal_param:
                    validated_proposals[param_name] = proposal_param
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Hyperparameter '{param_name}' is not recognized in the search space."
            )
            
    # Save to pending_search_space in DB
    try:
        with get_db_session() as session:
            # Check if there is already a pending configuration
            pending_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="pending_search_space"
            ).first()
            
            if validated_proposals:
                if pending_row:
                    # Merge with existing pending configuration
                    existing_pending = json.loads(pending_row.config_value)
                    for key, val in validated_proposals.items():
                        if key in existing_pending:
                            existing_pending[key].update(val)
                        else:
                            existing_pending[key] = val
                    pending_row.config_value = json.dumps(existing_pending)
                    pending_row.version += 1
                else:
                    session.add(SystemConfiguration(
                        study_name=study_name,
                        config_key="pending_search_space",
                        config_value=json.dumps(validated_proposals),
                        version=1
                    ))
            else:
                # If proposals are empty (reverted to current), delete pending row if it exists
                if pending_row:
                    session.delete(pending_row)
            session.commit()
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save pending changes: {str(e)}")
        
    return {"success": True, "space": current, "pending": validated_proposals}


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


@app.get("/api/studies")
def api_list_studies():
    try:
        summaries = optuna.get_all_study_summaries(storage=DATABASE_URL)
        return {"success": True, "studies": [s.study_name for s in summaries]}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
                "dice_eval_fixed": metrics["dice_eval_fixed"],
                "bce_eval_fixed": metrics["bce_eval_fixed"],
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
            importances = optuna.importance.get_param_importances(
                study,
                target=lambda t: t.values[1], # Maximize Dice
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

        est_imp = req.estimated_score_improvement if req.estimated_score_improvement is not None else req.estimated_dice_improvement
        validation = validate_review_fields(est_imp, req.cited_best_trial)
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
            estimated_dice_improvement=est_imp,
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


def _apply_search_space_patch(patch: Dict[str, Any], space: Dict[str, Any], study_name: str) -> str:
    """Validate + persist active-bound narrowing from a coordinator review."""
    for param, new_val in patch.items():
        if param not in space:
            return f"Unknown parameter '{param}'."
        cfg = space[param]
        if cfg.get("type") == "categorical":
            if "active" in new_val:
                allowed = cfg.get("options", [])
                invalid = [x for x in new_val["active"] if x not in allowed]
                if invalid:
                    return f"Active choices {invalid} for {param} not in options {allowed}."
                if not new_val["active"]:
                    return f"{param} must keep at least one active option."
                cfg["active"] = new_val["active"]
        else:
            if "min" in new_val:
                cfg["min"] = float(new_val["min"])
            if "max" in new_val:
                cfg["max"] = float(new_val["max"])
    save_search_space(space, study_name)
    mark_review_applied(study_name)
    return "Search space updated."

def _enqueue_manual_trial(study, manual: Dict[str, Any], space: Dict[str, Any], summary: str = "AI Coordinator suggested manual trial.") -> str:
    """Enqueue one coordinator-proposed trial; TPE still drives every other suggest."""
    config = load_hpo_config(study.study_name)
    params = normalize_trial_params(dict(manual), config)
    missing = [p for p in _expected_search_params(space) if p not in params]
    if missing:
        return f"Manual trial missing params {missing}; not enqueued."
    try:
        study.enqueue_trial(params)
        waiting = [t for t in study.trials if t.state == TrialState.WAITING]
        if waiting:
            new_trial = max(waiting, key=lambda t: t._trial_id)
            with get_db_session() as session:
                session.add(
                    AgentReasoningLog(
                        trial_id=new_trial._trial_id,
                        study_name=study.study_name,
                        model_version="coordinator",
                        prompt_strategy="coordinator_review",
                        predicted_outcome_rationale=summary,
                        estimated_score_improvement=0.0
                    )
                )
        return f"Enqueued manual trial: {params}."
    except Exception as e:
        return f"Could not enqueue manual trial: {e}"



# --- WORKER HPO ROUTER ENDPOINTS ---
@app.post("/api/suggest_trial")
@app.post("/api/suggest_trials")  # common typo alias
def api_suggest_trial(req: SuggestRequest):
    trial = None
    start_time = time.time()
    try:
        from datetime import datetime, timedelta

        # C2 Fix: thread session explicitly
        with get_db_session() as session:
            repaired = _repair_categorical_param_indices(session, req.study_name)
            if repaired:
                print(f"Repaired {repaired} corrupt categorical param row(s) in study '{req.study_name}'.")

        study = load_study(req.study_name)
        space = load_search_space(req.study_name)
        hpo_config = load_hpo_config(req.study_name)
        _cleanup_stuck_running_trials(study, space)

        # 1. Reclaim expired leases first
        with get_db_session() as session:
            _reap_expired_leases(study, req.study_name, session)

        # 2. Check for available RUNNING trials
        running_trials = [
            t
            for t in study.trials
            if t.state == TrialState.RUNNING
            and _trial_has_full_params(_worker_ready_params(t, space), space)
        ]

        source = "recycled_running"
        if running_trials:
            running_trials.sort(key=lambda t: t.number)
            leased_to = req.worker_id or "anonymous"
            for t in running_trials:
                # Atomic claim: at most one concurrent worker wins this trial (see _try_claim_lease).
                with get_db_session() as session:
                    claimed = _try_claim_lease(session, req.study_name, t._trial_id, leased_to)
                if claimed:
                    trial = t
                    break
            if trial:
                trial_id = trial._trial_id
                params = _worker_ready_params(trial, space)
                print(f"Assigning active leased RUNNING Trial {trial.number} to worker {req.worker_id}.")

        # 3. If no RUNNING trial is available, ask Optuna for a new one
        if not trial:
            source = "new_trial"
            _enqueue_single_active_categoricals(study, space)
            trial = study.ask()
            trial_id = trial._trial_id

            # Lease newly created trial immediately (fresh trial_id, so this always wins).
            with get_db_session() as session:
                _try_claim_lease(session, req.study_name, trial_id, req.worker_id or "anonymous")

            try:
                params = suggest_params_from_space(study, trial, space)
                _persist_fixed_categorical_params(study, trial, space)
            except Exception:
                try:
                    study.tell(trial.number, state=TrialState.FAIL)
                except Exception:
                    pass
                with get_db_session() as session:
                    session.query(TrialLease).filter_by(trial_id=trial_id).delete()
                    session.commit()
                raise

            with get_db_session() as session:
                existing = (
                    session.query(AgentReasoningLog).filter_by(trial_id=trial_id).first()
                )
                if not existing:
                    est_imp = req.estimated_score_improvement if req.estimated_score_improvement is not None else req.estimated_dice_improvement
                    session.add(
                        AgentReasoningLog(
                            trial_id=trial_id,
                            study_name=req.study_name,
                            model_version=req.agent_model or "optuna-tpe",
                            prompt_strategy=req.prompt_strategy or "tpe_sampler",
                            predicted_outcome_rationale=req.reasoning or "Autonomous worker suggestion request.",
                            estimated_score_improvement=float(est_imp if est_imp is not None else 0.0),
                        )
                    )
                    session.commit()

        params = _finalize_trial_params(params, space)
        missing = [p for p in _expected_search_params(space) if p not in params]
        if missing:
            try:
                study.tell(trial.number, state=TrialState.FAIL)
            except Exception:
                pass
            with get_db_session() as session:
                session.query(TrialLease).filter_by(trial_id=trial_id).delete()
                session.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Trial {trial.number} missing parameters {missing}. Expected {_expected_search_params(space)}.",
            )

        # Log SuggestMetric
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        try:
            with get_db_session() as session:
                session.add(SuggestMetric(
                    study_name=req.study_name,
                    latency_ms=latency_ms,
                    source=source
                ))
                session.commit()
        except Exception as metric_err:
            print(f"Error logging suggest metric: {metric_err}")

        config = hpo_config
        return {
            "success": True,
            "trial_id": trial_id,
            "trial_number": trial.number,
            "params": normalize_trial_params(params, config),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class HeartbeatRequest(BaseModel):
    study_name: str
    worker_id: str
    trial_id: int

@app.post("/api/heartbeat")
def api_heartbeat(req: HeartbeatRequest):
    try:
        from datetime import datetime, timedelta
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                study_name=req.study_name,
                trial_id=req.trial_id,
                leased_to=req.worker_id
            ).first()
            if lease:
                lease.lease_expires_at = datetime.utcnow() + timedelta(seconds=LEASE_TTL_SECONDS)
                session.commit()
                return {"success": True, "message": "Heartbeat acknowledged"}
            return {"success": False, "message": "Lease not found or expired"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _lease_is_owned(study_name: str, trial_id: int, worker_id: Optional[str]) -> bool:
    """True if a non-expired lease for this trial is held by ``worker_id``.

    Workers prove ownership with the ``worker_id`` returned from /api/suggest_trial before
    they may report epochs or complete an in-flight trial. Terminal trials skip this check
    (see callers): idempotent retries and post-prune completes must still record results even
    though the lease was already deleted. Note: compared against naive UTC because lease
    timestamps are stored via ``datetime.utcnow()``.
    """
    if not worker_id:
        return False
    from datetime import datetime
    with get_db_session() as session:
        lease = session.query(TrialLease).filter_by(
            study_name=study_name, trial_id=trial_id, leased_to=worker_id
        ).first()
        if lease is None:
            return False
        if lease.lease_expires_at is not None and lease.lease_expires_at < datetime.utcnow():
            return False
        return True


@app.post("/api/report_epoch")
def api_report_epoch(req: ReportEpochRequest):
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

        # Resolve final values (using generic if present, otherwise fallback to positional legacy)
        final_score = req.score if req.score is not None else req.dice
        final_loss = req.loss if req.loss is not None else req.bce
        final_score_fixed = req.score_eval_fixed if req.score_eval_fixed is not None else req.dice_eval_fixed
        final_loss_fixed = req.loss_eval_fixed if req.loss_eval_fixed is not None else req.bce_eval_fixed

        if final_score is None or final_loss is None:
            raise HTTPException(status_code=400, detail="Both score (or dice) and loss (or bce) must be provided.")

        # Save user attributes for real-time dashboard monitoring
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
        if final_loss_fixed is not None:
            study._storage.set_trial_user_attr(
                trial_obj._trial_id, ev.get("fixed_bce_attr", "bce_eval_fixed"), final_loss_fixed
            )

        history = list(trial_obj.user_attrs.get("history", []))
        history = [h for h in history if h.get("epoch") != req.epoch]
        epoch_entry = {"epoch": req.epoch, "dice": final_score, "bce": final_loss}
        if final_score_fixed is not None:
            epoch_entry["dice_eval_fixed"] = final_score_fixed
        if final_loss_fixed is not None:
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complete_trial")
def api_complete_trial(req: CompleteTrialRequest):
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

        # Resolve final values (using generic if present, otherwise fallback to positional legacy)
        final_score = req.score if req.score is not None else req.dice
        final_loss = req.loss if req.loss is not None else req.bce
        final_score_fixed = req.score_eval_fixed if req.score_eval_fixed is not None else req.dice_eval_fixed
        final_loss_fixed = req.loss_eval_fixed if req.loss_eval_fixed is not None else req.bce_eval_fixed

        if final_score is None or final_loss is None:
            raise HTTPException(status_code=400, detail="Both score (or dice) and loss (or bce) must be provided.")

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
                if final_loss_fixed is not None:
                    study._storage.set_trial_user_attr(
                        trial_obj._trial_id, ev.get("fixed_bce_attr", "bce_eval_fixed"), final_loss_fixed
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
                    study.tell(trial_obj.number, [final_loss, final_score])
                else:
                    study.tell(trial_obj.number, final_score)
            else:
                try:
                    study.tell(trial_obj.number, state=t_state)
                except Exception as tell_err:
                    print(f"Could not tell state to Optuna: {tell_err}")

        with get_db_session() as session:
            import math
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
            best_prior_dice = 0.0
            if prior_trials:
                if len(study.directions) > 1:
                    best_prior_dice = max([t.values[1] for t in prior_trials if t.values and len(t.values) > 1] or [0.0])
                else:
                    best_prior_dice = max([t.value for t in prior_trials if t.value is not None] or [0.0])

            actual_improvement = final_score - best_prior_dice
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

            from src.hpo_coordinator import write_ide_status_file
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
                if len(study.directions) > 1 and t.values and len(t.values) > 1:
                    completed_scores.append(t.values[1])
                elif t.value is not None:
                    completed_scores.append(t.value)
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tunnel_url")
def api_get_tunnel_url():
    """Returns the active ngrok tunnel URL if established."""
    try:
        with get_db_session() as session:
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
            "update_search_space",
            "submit_agent_review",
            "generate_model_card"
        ]
    }

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
    
    args = parser.parse_args()

    # Network safety: refuse to expose the broker beyond loopback (or via a tunnel) without a
    # token. This is the single most important guard for the tunneled threat model.
    tunnel_requested = args.tunnel or os.getenv("HPO_TUNNEL_ENABLED") == "1"
    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if (not is_loopback or tunnel_requested) and not os.environ.get("HPO_SECRET_TOKEN"):
        raise SystemExit(
            "Refusing to start: exposing the broker beyond loopback (non-loopback --host or "
            "--tunnel) without auth is unsafe. Set HPO_SECRET_TOKEN, or use --host 127.0.0.1 "
            "for local-only use."
        )
    
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
        
    # 2. Start ngrok tunnel if requested
    tunnel_enabled = args.tunnel or os.getenv("HPO_TUNNEL_ENABLED") == "1"
    if tunnel_enabled:
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
                                    
                                    # Persist in SQLite under SystemConfiguration so dashboard can fetch it
                                    try:
                                        from src.db_manager import get_db_session
                                        from src.schema import SystemConfiguration
                                        with get_db_session() as session:
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
