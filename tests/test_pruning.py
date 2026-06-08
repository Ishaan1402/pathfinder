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

from src.db_manager import init_db
from broker import _epoch_composite_score


class TestPruning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.study_name = "test_study_pruning_" + self._testMethodName
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=os.environ["HPO_DATABASE_URL"],
            directions=["minimize", "maximize"],
            load_if_exists=True
        )

    def _get_frozen_trial(self, trial_number):
        """Get the FrozenTrial object for a given trial number.

        _epoch_composite_score expects FrozenTrial (which has intermediate_values),
        not Trial (returned by study.ask()).
        """
        for t in self.study.trials:
            if t.number == trial_number:
                return t
        raise ValueError(f"Trial #{trial_number} not found")

    def test_composite_score_thin_data(self):
        """Less than 10 completed/running trials -> returns raw score (no Z-score)."""
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        t = self.study.ask()
        trial_number = t.number
        history = [{"epoch": 1, "dice": 0.5, "bce": 0.5}]
        self.study._storage.set_trial_user_attr(t._trial_id, "history", history)

        frozen = self._get_frozen_trial(trial_number)
        score = _epoch_composite_score(self.study, frozen, 1, {"enabled": False})
        self.assertEqual(score, 0.5)

    def test_composite_score_zscore_and_zero_variance_clamp(self):
        """11 trials with identical scores (zero variance) -> Z-score with epsilon clamp returns 0.0."""
        last_number = None
        for i in range(11):
            self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
            t = self.study.ask()
            self.study._storage.set_trial_user_attr(t._trial_id, "history", [{"epoch": 1, "dice": 0.8, "bce": 0.2}])
            if i < 10:
                self.study.tell(t.number, [0.2, 0.8])
            else:
                last_number = t.number

        frozen = self._get_frozen_trial(last_number)
        score = _epoch_composite_score(self.study, frozen, 1, {"enabled": False})
        # z_score = (0.8 - 0.8) / 1.0 = 0.0, z_loss = -(0.2 - 0.2) / 1.0 = 0.0
        self.assertEqual(score, 0.0)

    def test_composite_score_zscore_normal_variance(self):
        """11 trials with varying scores -> Z-score normalization produces meaningful non-zero result."""
        dice_scores = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        bce_losses =  [0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

        for i in range(10):
            self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
            t = self.study.ask()
            self.study._storage.set_trial_user_attr(
                t._trial_id, "history",
                [{"epoch": 1, "dice": dice_scores[i], "bce": bce_losses[i]}]
            )
            self.study.tell(t.number, [bce_losses[i], dice_scores[i]])

        # 11th trial — above-average (dice=0.9, bce=0.1)
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        outlier = self.study.ask()
        outlier_number = outlier.number
        self.study._storage.set_trial_user_attr(
            outlier._trial_id, "history",
            [{"epoch": 1, "dice": 0.9, "bce": 0.1}]
        )

        frozen_outlier = self._get_frozen_trial(outlier_number)
        score = _epoch_composite_score(self.study, frozen_outlier, 1, {"enabled": False})
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0, "Above-average trial should produce positive Z-score composite")

        # 12th trial — below-average (dice=0.2, bce=0.8)
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        weak = self.study.ask()
        weak_number = weak.number
        self.study._storage.set_trial_user_attr(
            weak._trial_id, "history",
            [{"epoch": 1, "dice": 0.2, "bce": 0.8}]
        )

        frozen_weak = self._get_frozen_trial(weak_number)
        weak_score = _epoch_composite_score(self.study, frozen_weak, 1, {"enabled": False})
        self.assertIsNotNone(weak_score)
        self.assertLess(weak_score, 0.0, "Below-average trial should produce negative Z-score composite")


if __name__ == "__main__":
    unittest.main()
