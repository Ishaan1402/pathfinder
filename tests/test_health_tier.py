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
import optuna
from optuna.trial import TrialState

from src.db_manager import init_db, get_db_session
from src.schema import TrialResult, StudyStatus
from src.hpo_coordinator import compute_health_tier


class TestHealthTier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.study_name = "test_study_health_" + self._testMethodName
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=os.environ["HPO_DATABASE_URL"],
            directions=["minimize", "maximize"],
            load_if_exists=True
        )

    def test_healthy_study(self):
        """Empty/fresh study returns 'healthy'."""
        tier, reason = compute_health_tier(self.study, self.study_name)
        self.assertEqual(tier, "healthy")

    def test_inf_metrics(self):
        """Inf in reported metrics triggers 'intervene' with NaN or Inf detection."""
        # Optuna rejects NaN (converts to FAIL with values=None), but accepts Inf
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        trial = self.study.ask()
        self.study.tell(trial.number, [float('inf'), 0.5])

        tier, reason = compute_health_tier(self.study, self.study_name)
        self.assertEqual(tier, "intervene")
        self.assertIn("NaN or Inf detected", reason)

    def test_oom_cluster(self):
        """Same parameter combination causing 2+ OOM failures triggers 'intervene'."""
        fixed_params = {"learning_rate": 1e-3, "batch_size": 16}

        self.study.enqueue_trial(fixed_params)
        t1 = self.study.ask()
        t1.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        t1.suggest_categorical("batch_size", [8, 16, 32])
        self.study.tell(t1.number, state=TrialState.FAIL)

        self.study.enqueue_trial(fixed_params)
        t2 = self.study.ask()
        t2.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        t2.suggest_categorical("batch_size", [8, 16, 32])
        self.study.tell(t2.number, state=TrialState.FAIL)

        # Verify params are populated
        for t in self.study.trials:
            self.assertTrue(len(t.params) > 0, f"Trial #{t.number} should have params")

        with get_db_session() as session:
            session.add(TrialResult(
                trial_id=t1._trial_id,
                study_name=self.study_name,
                epoch_reached=1,
                oom_triggered=True
            ))
            session.add(TrialResult(
                trial_id=t2._trial_id,
                study_name=self.study_name,
                epoch_reached=1,
                oom_triggered=True
            ))
            session.commit()

        tier, reason = compute_health_tier(self.study, self.study_name)
        self.assertEqual(tier, "intervene")
        self.assertIn("OOM cluster detected", reason)

    def test_stagnation(self):
        """No score improvement over many completed trials triggers 'intervene'."""
        scores = [0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        for score in scores:
            self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
            t = self.study.ask()
            self.study.tell(t.number, [2.0 - score, score])

        tier, reason = compute_health_tier(self.study, self.study_name)
        self.assertEqual(tier, "intervene")
        self.assertIn("Score stagnation", reason)


if __name__ == "__main__":
    unittest.main()
