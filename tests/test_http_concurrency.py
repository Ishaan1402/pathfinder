"""Concurrency test for the atomic lease claim that prevents double-assigning a trial.

The broker's suggest path delegates the race-critical step to ``broker._try_claim_lease``.
We test that primitive directly with many threads contending for the SAME trial_id and assert
exactly one wins — deterministic and independent of the (single-threaded) Starlette TestClient.
Real end-to-end HTTP concurrency is additionally exercised by tests/test_integration.py.
"""
import threading
from datetime import datetime, timedelta


def _claim_in_new_session(study_name, trial_id, worker_id):
    from src.leases import _try_claim_lease
    from src.db_manager import get_db_session
    with get_db_session() as session:
        return _try_claim_lease(session, study_name, trial_id, worker_id)


def test_atomic_lease_claim_has_single_winner(initialized_study):
    trial_id = 990001  # fixed id with no pre-existing lease
    n_workers = 12
    winners = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_workers)

    def contend(wid):
        barrier.wait()  # release all threads simultaneously to maximize contention
        won = _claim_in_new_session(initialized_study, trial_id, f"worker-{wid}")
        if won:
            with lock:
                winners.append(wid)

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"Exactly one worker must win the lease, got {len(winners)}: {winners}"


def test_expired_lease_can_be_reclaimed_by_new_worker(initialized_study):
    trial_id = 990002
    assert _claim_in_new_session(initialized_study, trial_id, "owner-A") is True
    # A second worker cannot steal a live lease.
    assert _claim_in_new_session(initialized_study, trial_id, "owner-B") is False

    # Expire the lease, then a new worker may reclaim it.
    from src.db_manager import get_db_session
    from src.schema import TrialLease
    with get_db_session() as session:
        lease = session.query(TrialLease).filter_by(trial_id=trial_id).first()
        lease.lease_expires_at = datetime.utcnow() - timedelta(seconds=30)
    assert _claim_in_new_session(initialized_study, trial_id, "owner-B") is True
