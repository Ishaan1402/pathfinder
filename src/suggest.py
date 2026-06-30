
import json
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException
import optuna
from optuna.trial import TrialState
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db_manager import get_db_session, DATABASE_URL
from src.hpo_config import load_hpo_config, normalize_trial_params
from src.search_space import (
    load_search_space,
    _cleanup_stuck_running_trials,
    _trial_has_full_params,
    _worker_ready_params,
    _enqueue_single_active_categoricals,
    suggest_params_from_space,
    _finalize_trial_params,
    _expected_search_params,
)
from src.leases import _reap_expired_leases, _try_claim_lease, delete_lease_by_trial_id


def get_or_create_study(study_name: str):
    try:
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    except KeyError:
        print(f"Study '{study_name}' not found. Checking stored config for directions...")
        directions = None
        try:
            cfg = load_hpo_config(study_name)
            directions = cfg.get("directions")
        except Exception:
            pass
        if not directions:
            directions = ["minimize", "maximize"]
            print("No stored directions found. Defaulting to multi-objective (minimize, maximize).")
        return optuna.create_study(
            study_name=study_name,
            storage=DATABASE_URL,
            directions=directions,
            load_if_exists=True
        )


def load_study(study_name: str):
    """Load an existing Optuna study, or raise 404 if it was never initialized."""
    try:
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Study '{study_name}' is not initialized. Create it first via the Dashboard, CLI init, or MCP init_from_manifest tool.",
        )


class SuggestRequest(BaseModel):
    study_name: str
    worker_id: Optional[str] = None
    agent_model: Optional[str] = "optuna-tpe"
    prompt_strategy: Optional[str] = "tpe_sampler"
    reasoning: Optional[str] = "Autonomous worker suggestion request."
    estimated_score_improvement: Optional[float] = None


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
            # Determine if param_value is stored as an internal index or an external value.
            # If it can be parsed as a float, check whether it matches a choice BY VALUE first.
            try:
                external = float(param_value)
            except (ValueError, TypeError):
                external = param_value
            # If the stored value equals one of the choices literally, treat it as an
            # external value and convert to its internal index. Otherwise, assume it is
            # already a valid internal index.
            if external in choices:
                internal = choices.index(external)
            elif isinstance(external, float) and int(external) in choices:
                internal = choices.index(int(external))
            else:
                # Already an internal index or unrecognised; skip repair.
                idx = int(float(param_value))
                if 0 <= idx < len(choices):
                    continue
                continue
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


def handle_api_suggest_trial(req: SuggestRequest):
    trial = None
    try:
        # Load the study first to verify it exists. This raises a clean 404 if the study is uninitialized.
        study = load_study(req.study_name)

        # C2 Fix: thread session explicitly
        with get_db_session() as session:
            try:
                repaired = _repair_categorical_param_indices(session, req.study_name)
                if repaired:
                    print(f"Repaired {repaired} corrupt categorical param row(s) in study '{req.study_name}'.")
            except Exception as e:
                # Fallback: if there is a database issue during repair, do not crash suggestion
                import logging
                logging.getLogger(__name__).warning(f"Failed to repair categorical param indices: {e}")
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
            _enqueue_single_active_categoricals(study, space)

            # Optuna does not support narrowing categorical distributions after the first
            # trial.  TPE always samples from the full historical choice set.  When a user
            # deactivates a choice (dashboard Settings), we must handle TPE sampling an
            # inactive value.  We retry up to 20 times, failing each attempt in Optuna.
            # This tells TPE that the deactivated region is unproductive — a best-effort
            # heuristic given Optuna's static-distribution design.  On the final attempt,
            # we substitute a random active choice instead of 500-ing the worker.
            MAX_RESAMPLE_ATTEMPTS = 20
            for attempt in range(MAX_RESAMPLE_ATTEMPTS):
                trial = study.ask()
                trial_id = trial._trial_id

                with get_db_session() as session:
                    claimed = _try_claim_lease(session, req.study_name, trial_id, req.worker_id or "anonymous")
                    assert claimed, f"Lease claim failed for trial {trial_id}"

                try:
                    params = suggest_params_from_space(study, trial, space)
                    break
                except ValueError:
                    # TPE sampled an inactive categorical
                    if attempt < MAX_RESAMPLE_ATTEMPTS - 1:
                        # Fail the trial and retry with a fresh one
                        try:
                            study.tell(trial.number, state=TrialState.FAIL)
                        except Exception:
                            pass
                        with get_db_session() as session:
                            delete_lease_by_trial_id(session, trial_id)
                            session.commit()
                        continue
                    # Final attempt — fallback: pick random active values for violated categoricals
                    import random
                    fixed = _finalize_trial_params(dict(trial.params), space)
                    for param, cfg in space.items():
                        if isinstance(cfg, dict) and cfg.get("type") == "categorical":
                            active = list(cfg.get("active") or cfg.get("options") or [])
                            if active and fixed.get(param) not in active:
                                fallback_choice = random.choice(active)
                                fixed[param] = fallback_choice
                                print(
                                    f"WARNING: Trial {trial.number} — {param} "
                                    f"outside active {active}. Falling back to {fallback_choice!r}."
                                )
                    params = fixed
            
        params = _finalize_trial_params(params, space)
        missing = [p for p in _expected_search_params(space) if p not in params]
        if missing:
            try:
                study.tell(trial.number, state=TrialState.FAIL)
            except Exception:
                pass
            with get_db_session() as session:
                delete_lease_by_trial_id(session, trial_id)
                session.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Trial {trial.number} missing parameters {missing}. Expected {_expected_search_params(space)}.",
            )

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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


