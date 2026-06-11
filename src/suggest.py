import time
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from fastapi import HTTPException
import optuna
from optuna.trial import TrialState
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db_manager import get_db_session, DATABASE_URL
from src.schema import AgentReasoningLog, SuggestMetric, TrialLease
from src.hpo_config import load_hpo_config, normalize_trial_params
from src.search_space import (
    load_search_space,
    _cleanup_stuck_running_trials,
    _trial_has_full_params,
    _worker_ready_params,
    _enqueue_single_active_categoricals,
    suggest_params_from_space,
    _persist_fixed_categorical_params,
    _finalize_trial_params,
    _expected_search_params,
)
from src.leases import _reap_expired_leases, _try_claim_lease


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
    """Load an existing Optuna study, or raise 404 if it was never initialized."""
    try:
        return optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Study '{study_name}' is not initialized. Create it first via the MCP initialize_study tool.",
        )


class SuggestRequest(BaseModel):
    study_name: str
    worker_id: Optional[str] = None
    agent_model: Optional[str] = "optuna-tpe"
    prompt_strategy: Optional[str] = "tpe_sampler"
    reasoning: Optional[str] = "Autonomous worker suggestion request."
    estimated_score_improvement: Optional[float] = None
    estimated_dice_improvement: Optional[float] = 0.0


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


def handle_api_suggest_trial(req: SuggestRequest):
    trial = None
    start_time = time.time()
    try:
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
