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
from src.schema import TrialResult, StudyReview
from src.hpo_coordinator import build_review_packet, build_review_prompt, save_study_review


class TestCoordinatorPacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.study_name = "test_study_coord_" + self._testMethodName
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=os.environ["HPO_DATABASE_URL"],
            directions=["minimize", "maximize"],
            load_if_exists=True
        )

    def test_build_packet_and_prompt(self):
        """Review prompt contains best trial info and packet has required keys."""
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        t = self.study.ask()
        self.study.tell(t.number, [0.2, 0.8])

        # Test review prompt references the best trial
        prompt = build_review_prompt(self.study_name)
        self.assertIn("Trial #0", prompt)
        self.assertIn("0.8000", prompt)

        # Test build review packet has binned trials and keys
        packet = build_review_packet(self.study_name)
        self.assertIn("trial_bins", packet)
        self.assertIn("spearman_correlations", packet)

    def test_save_study_review_confidence_low(self):
        """Citing a non-existent trial results in low confidence."""
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        t = self.study.ask()
        self.study.tell(t.number, [0.1, 0.9])

        res = save_study_review(
            self.study_name,
            "Citing wrong trial number",
            health_rating=3,
            policy_action="no_change",
            trials_evaluated=1,
            cited_best_trial=999  # non-existent trial
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["review"]["confidence"], "low")

    def test_save_study_review_confidence_high(self):
        """Citing the correct best trial results in high confidence."""
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 16})
        t = self.study.ask()
        self.study.tell(t.number, [0.1, 0.9])

        res = save_study_review(
            self.study_name,
            "Citing correct trial number",
            health_rating=4,
            policy_action="no_change",
            trials_evaluated=1,
            cited_best_trial=0
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["review"]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
