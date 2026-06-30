import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException
from optuna.trial import TrialState
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from src.db_manager import get_db_session
from src.schema import TrialLease

# ~3× worker heartbeat interval (15s in hpo_client). Clears ghost RUNNING rows soon after crash.
LEASE_TTL_SECONDS = 45


def _reap_stale_running_trials(study, study_name: str, session) -> int:
    """Fail RUNNING trials whose worker lease expired or has no active lease."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
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
        # Skip trials that were created very recently (no lease row yet) to avoid
        # reaping a trial that a concurrent suggest call is still setting up.
        if t._trial_id not in active_leased_ids and t._trial_id not in expired_trial_ids:
            if t.datetime_start is not None:
                age = (now - t.datetime_start.replace(tzinfo=None)).total_seconds()
                if age < LEASE_TTL_SECONDS:
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
    now = datetime.now(timezone.utc).replace(tzinfo=None)
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


def _lease_is_owned(study_name: str, trial_id: int, worker_id: Optional[str]) -> bool:
    """True if a non-expired lease for this trial is held by ``worker_id``.

    Workers prove ownership with the ``worker_id`` returned from /api/suggest_trial before
    they may report epochs or complete an in-flight trial. Terminal trials skip this check
    (see callers): idempotent retries and post-prune completes must still record results even
    though the lease was already deleted. Lease timestamps are stored as naive UTC.
    """
    if not worker_id:
        return False
    with get_db_session() as session:
        lease = session.query(TrialLease).filter_by(
            study_name=study_name, trial_id=trial_id, leased_to=worker_id
        ).first()
        if lease is None:
            return False
        if lease.lease_expires_at is not None and lease.lease_expires_at < datetime.now(timezone.utc).replace(tzinfo=None):
            return False
        return True


class HeartbeatRequest(BaseModel):
    study_name: str
    worker_id: str
    trial_id: int


def handle_api_heartbeat(req: HeartbeatRequest):
    try:
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                study_name=req.study_name,
                trial_id=req.trial_id,
                leased_to=req.worker_id
            ).first()
            if lease:
                lease.lease_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=LEASE_TTL_SECONDS)
                session.commit()
                return {"success": True, "message": "Heartbeat acknowledged"}
            return {"success": False, "message": "Lease not found or expired"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def delete_lease_by_trial_id(session, trial_id: int) -> None:
    session.query(TrialLease).filter_by(trial_id=trial_id).delete()
