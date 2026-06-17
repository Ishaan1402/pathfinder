"""Background polling daemon for monitoring study health.

Runs inside the FastAPI broker process as a background thread. It is intentionally NOT an
autopilot: it reclaims expired trial leases, recomputes each study's health tier, writes the
``.hpo_status.json`` hint file, and (optionally) fires a desktop notification so the human can
open the dashboard. It never calls an LLM and never mutates the search space — coordinator
reviews are always human-initiated (dashboard "Apply Proposal" or the MCP tools).
"""
import time
import subprocess
from typing import Dict

from src.db_manager import get_db_session
from src.schema import StudyStatus
from src.hpo_coordinator import compute_review_heuristics
from src.hpo_config import load_hpo_config

DEFAULT_STUDY = "bridge_crack_study"

# Cooldown to avoid alert spamming: {study_name: (last_alert_time, last_alert_trials)}
_ALERT_COOLDOWN_PERIOD = 300  # 5 minutes
_alert_history: Dict[str, tuple] = {}


def trigger_macos_notification(title: str, subtitle: str, message: str):
    """Triggers a native macOS desktop alert popup using AppleScript."""
    try:
        title_esc = title.replace('"', '\\"')
        subtitle_esc = subtitle.replace('"', '\\"')
        msg_esc = message.replace('"', '\\"')

        script = f'display notification "{msg_esc}" with title "{title_esc}" subtitle "{subtitle_esc}"'
        subprocess.run(["osascript", "-e", script], check=True)
    except Exception as e:
        print(f"Error triggering macOS desktop notification: {e}")


def check_and_alert_study(study_name: str):
    """Recompute a study's health, refresh the status hint file, and notify if warranted.

    Notify-only: when a review is recommended and desktop notifications are enabled, fire a
    macOS notification pointing the user at the dashboard. No LLM is ever called here.
    """
    try:
        from .suggest import get_or_create_study

        try:
            study = get_or_create_study(study_name)
        except Exception:
            # Study might not exist yet, skip silently
            return

        hpo_config = load_hpo_config(study_name)
        notifs_enabled = hpo_config.get("desktop_notifications_enabled", False)

        from src.hpo_coordinator import study_eval_insights, write_ide_status_file
        insights = study_eval_insights(study, hpo_config)
        heuristics = compute_review_heuristics(study, insights, hpo_config, study_name)

        n_eval = heuristics["trials_evaluated"]
        health_tier = heuristics["health_tier"]
        health_reason = heuristics["health_reason"]

        # Refresh the IDE status hint file (informational; agents may read it on demand).
        write_ide_status_file(study_name, health_tier, health_reason, study)

        if not heuristics["review_recommended"]:
            # Healthy, or already reviewed/dismissed for the current trial window.
            return

        # Cooldown to avoid repeat alerts for the same trial window.
        now = time.time()
        if study_name in _alert_history:
            last_time, last_trials = _alert_history[study_name]
            if last_trials == n_eval or (now - last_time) < _ALERT_COOLDOWN_PERIOD:
                return
        _alert_history[study_name] = (now, n_eval)

        print(f"⚠️ Pathfinder Alert [{study_name.upper()}]: {health_tier.upper()} state. Reason: {health_reason}")
        if notifs_enabled:
            subtitle = f"Study Health: {health_tier.upper()}"
            trigger_macos_notification("Pathfinder", subtitle, f"Trial #{n_eval} | {health_reason} (Open dashboard to review)")

    except Exception as e:
        print(f"Error checking study health: {e}")


def reclaim_expired_leases():
    from datetime import datetime
    import optuna
    from src.schema import TrialLease
    from .suggest import get_or_create_study

    try:
        with get_db_session() as session:
            now = datetime.utcnow()
            expired = session.query(TrialLease).filter(
                TrialLease.lease_expires_at < now
            ).all()
            if expired:
                for lease in expired:
                    try:
                        study = get_or_create_study(lease.study_name)
                        # Find corresponding trial number in Optuna
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
                                print(
                                    f"Daemon: Terminated expired leased Trial {trial_number} "
                                    f"(ID {lease.trial_id}) in study '{lease.study_name}'."
                                )
                    except Exception as e:
                        print(f"Daemon: Failed to cleanly terminate expired trial ID {lease.trial_id}: {e}")
                    session.delete(lease)
                session.commit()
    except Exception as e:
        print(f"Daemon: Error in reclaim_expired_leases: {e}")


def run_daemon_loop(interval_seconds: int = 10):
    """Indefinite daemon polling loop (notify-only health monitoring + lease reclamation)."""
    print(f"Starting background health daemon thread (notify-only, interval: {interval_seconds}s)...")
    last_reap_time = 0.0
    while True:
        try:
            now = time.time()
            if now - last_reap_time >= 60.0:
                reclaim_expired_leases()
                last_reap_time = now

            studies = []
            try:
                with get_db_session() as session:
                    rows = session.query(StudyStatus).all()
                    studies = [r.study_name for r in rows]
            except Exception as e:
                print(f"Failed to fetch studies: {e}")

            if not studies:
                studies = [DEFAULT_STUDY]

            for name in studies:
                check_and_alert_study(name)

        except Exception as e:
            print(f"Daemon loop error: {e}")

        time.sleep(interval_seconds)
