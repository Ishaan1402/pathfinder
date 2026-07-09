
import json
import logging
from typing import Optional, Dict, Any, List

from fastapi import HTTPException
from optuna.trial import TrialState

from src.db_manager import get_db_session
from src.schema import SystemConfiguration, CompactedPacket

logger = logging.getLogger(__name__)

# Default search space definition
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float_log"},
    "batch_size": {"options": [16, 32, 64, 128], "active": [16, 32, 64, 128], "type": "categorical"},
}


def load_search_space(study_name: Optional[str] = None) -> Dict[str, Any]:
    from .settings import settings
    if not study_name:
        study_name = settings.study_name
    if not study_name or not study_name.strip():
        return DEFAULT_SEARCH_SPACE.copy()

    try:
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="active_search_space"
            ).first()
            if row:
                space = json.loads(row.config_value)
                return space
    except Exception as e:
        print(f"Error loading search space from DB: {e}")

    # Seed it in DB if not found
    try:
        save_search_space(DEFAULT_SEARCH_SPACE, study_name)
    except Exception as e:
        logger.warning(f"Failed to seed search_space in DB: {e}")
    return DEFAULT_SEARCH_SPACE.copy()


def save_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    from .settings import settings
    if not study_name:
        study_name = settings.study_name
    if not study_name or not study_name.strip():
        raise ValueError("study_name cannot be empty.")
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

    # Bust cached study packets — search space changes invalidate analytics
    try:
        with get_db_session() as session:
            session.query(CompactedPacket).filter_by(study_name=study_name).delete()
    except Exception:
        pass


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
    """Fail RUNNING trials that never received a full parameter set (crashed mid-suggest).

    Trials that started less than LEASE_TTL_SECONDS ago are skipped to avoid a race
    where worker A's trial was just created by ``study.ask()`` but hasn't yet had
    its parameters written when worker B's ``/api/suggest_trial`` triggers cleanup.
    """
    from .leases import LEASE_TTL_SECONDS
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=LEASE_TTL_SECONDS)
    for t in list(study.trials):
        if t.state != TrialState.RUNNING:
            continue
        if not _trial_has_full_params(_worker_ready_params(t, space), space):
            if t.datetime_start is not None and t.datetime_start.replace(tzinfo=None) > cutoff:
                continue
            print(f"Failing stuck RUNNING trial #{t.number} (incomplete params: {list(t.params.keys())})")
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


def _validate_categorical_against_active(params: Dict[str, Any], space: Dict[str, Any]) -> List[str]:
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
        elif ptype == "int":
            trial.suggest_int(param, int(cfg["min"]), int(cfg["max"]))
        elif ptype == "categorical":
            _suggest_categorical_compatible(study, trial, param, cfg)
        else:
            raise ValueError(f"Unsupported parameter type '{ptype}' for '{param}'.")
    params = _finalize_trial_params(trial.params, space)
    violations = _validate_categorical_against_active(params, space)
    if violations:
        raise ValueError(
            "Sampled parameters outside active search bounds: "
            + "; ".join(violations)
            + ". Start a new study or widen active options."
        )
    return params


def _apply_search_space_patch(patch: Dict[str, Any], space: Dict[str, Any], study_name: str) -> str:
    """Validate + persist active-bound narrowing from a search-space patch."""
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
    return "Search space updated."


def handle_api_get_search_space(study_name: Optional[str] = None):
    return load_search_space(study_name)


def handle_api_update_search_space(space: Dict[str, Any], study_name: Optional[str] = None):
    from .settings import settings
    if not study_name:
        study_name = space.get("study_name") or settings.study_name
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
            
    # Apply validated proposals directly to active search space
    if validated_proposals:
        try:
            for param_name, proposal in validated_proposals.items():
                current[param_name].update(proposal)
            # Validate min < max for all numeric params after merge
            for param_name, cfg in current.items():
                if not isinstance(cfg, dict) or cfg.get("type") not in ("float", "float_log", "int"):
                    continue
                lo = cfg.get("min")
                hi = cfg.get("max")
                if lo is not None and hi is not None and float(lo) >= float(hi):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid bounds for '{param_name}': min ({lo}) must be strictly less than max ({hi}).",
                    )
            save_search_space(current, study_name)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update search space: {str(e)}")

    return {"success": True, "space": current}
