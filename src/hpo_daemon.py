"""Background polling daemon for study health monitoring only.

Runs inside the FastAPI broker process as a background thread. It reclaims expired trial
leases, recomputes each study's health tier via ``compute_health_tier``, writes the
``.hpo_status.json`` hint file, and updates ``StudyStatus`` in the database. It never calls
an LLM, never fires desktop notifications, and never recommends coordinator reviews.
"""
import logging
import time
from typing import List

from src.db_manager import get_db_session
from src.health import compute_health_tier, write_ide_status_file
from src.schema import StudyStatus

logger = logging.getLogger(__name__)


def check_and_alert_study(study_name: str):
    """Recompute a study's health tier, write the IDE status file, and persist to DB."""
    try:
        from .suggest import load_study

        try:
            study = load_study(study_name)
        except Exception:
            return

        health_tier, health_reason = compute_health_tier(study, study_name)

        write_ide_status_file(study_name, health_tier, health_reason, study)

        with get_db_session() as session:
            status = session.query(StudyStatus).filter_by(study_name=study_name).first()
            if status is None:
                status = StudyStatus(study_name=study_name)
                session.add(status)
            status.health_tier = health_tier
            status.health_reason = health_reason
            session.commit()

    except Exception:
        logger.exception("Error checking health for study %s", study_name)


def reclaim_expired_leases():
    from datetime import datetime, timezone
    import optuna
    from src.schema import TrialLease
    from .suggest import load_study

    try:
        with get_db_session() as session:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            expired = session.query(TrialLease).filter(
                TrialLease.lease_expires_at < now
            ).all()
            if expired:
                for lease in expired:
                    try:
                        study = load_study(lease.study_name)
                        trial_number = None
                        for t in study.trials:
                            if t._trial_id == lease.trial_id:
                                trial_number = t.number
                                break
                        if trial_number is not None:
                            trial_obj = next(
                                (t for t in study.trials if t.number == trial_number), None
                            )
                            if trial_obj and trial_obj.state == optuna.trial.TrialState.RUNNING:
                                study.tell(trial_number, state=optuna.trial.TrialState.FAIL)
                                logger.info(
                                    "Terminated expired leased Trial %d (ID %s) in study '%s'.",
                                    trial_number, lease.trial_id, lease.study_name,
                                )
                    except Exception:
                        logger.exception(
                            "Failed to cleanly terminate expired trial ID %s", lease.trial_id
                        )
                    session.delete(lease)
                session.commit()
    except Exception:
        logger.exception("Error in reclaim_expired_leases")


def run_daemon_loop(interval_seconds: int = 10):
    """Indefinite daemon polling loop (health monitoring + lease reclamation)."""
    logger.info("Starting background health daemon thread (interval: %ds)...", interval_seconds)
    last_reap_time = 0.0
    while True:
        try:
            now = time.time()
            if now - last_reap_time >= 60.0:
                reclaim_expired_leases()
                last_reap_time = now

            studies: List[str] = []
            try:
                with get_db_session() as session:
                    rows = session.query(StudyStatus).all()
                    studies = [r.study_name for r in rows]
            except Exception:
                logger.exception("Failed to fetch studies")

            for name in studies:
                check_and_alert_study(name)

        except Exception:
            logger.exception("Daemon loop error")

        time.sleep(interval_seconds)
