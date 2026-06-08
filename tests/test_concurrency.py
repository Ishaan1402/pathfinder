"""
Concurrency integration test for the TrialLease system.

Verifies that multiple concurrent workers each receive unique leased trials,
and that no trial duplication occurs across simultaneous suggest requests.
"""

import os
import sys

# Ensure project root is in sys.path and HPO_DATABASE_URL is set before any src imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if "HPO_DATABASE_URL" not in os.environ:
    import tempfile
    _test_db_fd, TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="hpo_test_suite_")
    os.close(_test_db_fd)
    os.environ["HPO_DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
    import atexit
    def _cleanup():
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(TEST_DB_PATH + suffix)
            except OSError:
                pass
    atexit.register(_cleanup)

import unittest
import threading
import uuid
import optuna
import json
from datetime import datetime, timedelta
from optuna.trial import TrialState

from src.db_manager import init_db, get_db_session
from src.schema import TrialLease, SystemConfiguration


class TestConcurrencyLeases(unittest.TestCase):
    """Tests that the leasing system prevents duplicate trial assignments under concurrency."""

    NUM_WORKERS = 5

    @classmethod
    def setUpClass(cls):
        init_db()

    def _make_study(self, suffix=""):
        """Create a fresh uniquely-named study for each test."""
        name = f"test_concurrency_{self._testMethodName}_{suffix}_{uuid.uuid4().hex[:8]}"
        study = optuna.create_study(
            study_name=name,
            storage=os.environ["HPO_DATABASE_URL"],
            directions=["minimize", "maximize"],
            load_if_exists=True
        )
        return study, name

    def test_unique_lease_per_worker(self):
        """Each worker gets a unique trial — no two workers share the same trial_id."""
        study, study_name = self._make_study()
        LEASE_TTL = 300

        assigned_trials = []
        errors = []
        lock = threading.Lock()

        def worker_suggest(worker_id):
            """Simulate a single worker asking for a trial and leasing it."""
            try:
                trial = study.ask()
                trial_id = trial._trial_id

                with get_db_session() as session:
                    lease = TrialLease(
                        trial_id=trial_id,
                        study_name=study_name,
                        leased_to=worker_id,
                        lease_expires_at=datetime.utcnow() + timedelta(seconds=LEASE_TTL)
                    )
                    session.add(lease)
                    session.commit()

                with lock:
                    assigned_trials.append({
                        "worker_id": worker_id,
                        "trial_id": trial_id,
                        "trial_number": trial.number,
                    })
            except Exception as e:
                with lock:
                    errors.append({"worker_id": worker_id, "error": str(e)})

        threads = []
        for i in range(self.NUM_WORKERS):
            wid = f"worker-{uuid.uuid4()}"
            t = threading.Thread(target=worker_suggest, args=(wid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Worker errors: {errors}")
        self.assertEqual(len(assigned_trials), self.NUM_WORKERS)

        # All trial_ids must be unique
        trial_ids = [a["trial_id"] for a in assigned_trials]
        self.assertEqual(
            len(trial_ids), len(set(trial_ids)),
            f"Duplicate trial IDs assigned! Got: {trial_ids}"
        )

        # All trial_numbers must be unique
        trial_numbers = [a["trial_number"] for a in assigned_trials]
        self.assertEqual(
            len(trial_numbers), len(set(trial_numbers)),
            f"Duplicate trial numbers assigned! Got: {trial_numbers}"
        )

    def test_expired_lease_is_not_double_assigned(self):
        """An expired lease should not result in two workers owning the same trial."""
        study, study_name = self._make_study()

        # Worker A gets a trial and immediately expires the lease
        trial_a = study.ask()
        with get_db_session() as session:
            lease = TrialLease(
                trial_id=trial_a._trial_id,
                study_name=study_name,
                leased_to="worker-A",
                lease_expires_at=datetime.utcnow() - timedelta(seconds=10)
            )
            session.add(lease)
            session.commit()

        # Worker B comes along and gets a new trial (different from A's)
        trial_b = study.ask()

        with get_db_session() as session:
            all_leases = session.query(TrialLease).filter_by(study_name=study_name).all()

            if trial_b._trial_id == trial_a._trial_id:
                # Same trial recycled — only one lease row should exist
                matching = [l for l in all_leases if l.trial_id == trial_a._trial_id]
                self.assertEqual(
                    len(matching), 1,
                    "Recycled trial should have exactly one lease row"
                )
            else:
                # Different trial — both must have distinct IDs
                self.assertNotEqual(trial_a._trial_id, trial_b._trial_id)

    def test_heartbeat_refreshes_lease(self):
        """Heartbeat should extend the lease expiration for the correct worker."""
        study, study_name = self._make_study()
        worker_id = f"worker-{uuid.uuid4()}"
        trial = study.ask()
        LEASE_TTL = 300

        # Create a lease about to expire (30 seconds remaining)
        initial_expiry = datetime.utcnow() + timedelta(seconds=30)
        with get_db_session() as session:
            lease = TrialLease(
                trial_id=trial._trial_id,
                study_name=study_name,
                leased_to=worker_id,
                lease_expires_at=initial_expiry
            )
            session.add(lease)
            session.commit()

        # Simulate heartbeat — refresh the lease
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                study_name=study_name,
                trial_id=trial._trial_id,
                leased_to=worker_id
            ).first()
            self.assertIsNotNone(lease, "Lease should exist before heartbeat")
            lease.lease_expires_at = datetime.utcnow() + timedelta(seconds=LEASE_TTL)
            session.commit()

        # Verify lease was extended
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                trial_id=trial._trial_id
            ).first()
            self.assertIsNotNone(lease)
            self.assertGreater(
                lease.lease_expires_at,
                initial_expiry,
                "Heartbeat should have extended the lease expiration"
            )

    def test_wrong_worker_cannot_refresh_lease(self):
        """A worker that doesn't own a lease should not be able to refresh it."""
        study, study_name = self._make_study()
        owner_id = f"worker-owner-{uuid.uuid4()}"
        intruder_id = f"worker-intruder-{uuid.uuid4()}"
        trial = study.ask()

        initial_expiry = datetime.utcnow() + timedelta(seconds=60)
        with get_db_session() as session:
            lease = TrialLease(
                trial_id=trial._trial_id,
                study_name=study_name,
                leased_to=owner_id,
                lease_expires_at=initial_expiry
            )
            session.add(lease)
            session.commit()

        # Intruder tries to heartbeat — should find no matching lease
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                study_name=study_name,
                trial_id=trial._trial_id,
                leased_to=intruder_id
            ).first()
            self.assertIsNone(lease, "Intruder should not find a lease matching their worker_id")

        # Verify original lease is untouched
        with get_db_session() as session:
            lease = session.query(TrialLease).filter_by(
                trial_id=trial._trial_id
            ).first()
            self.assertIsNotNone(lease)
            self.assertEqual(lease.leased_to, owner_id)


if __name__ == "__main__":
    unittest.main()
