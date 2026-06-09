import os
import sys
import datetime
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if "HPO_DATABASE_URL" not in os.environ:
    import tempfile
    _test_db_fd, TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="hpo_lean_")
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

import optuna
from optuna.trial import TrialState

from src.db_manager import init_db, get_db_session
from src.schema import StudyReview
from src.hpo_coordinator import (
    compute_statistical_confidence,
    compute_coordinator_accuracy,
    backfill_review_outcomes,
    mark_review_applied,
    validate_review_fields,
    save_study_review,
    build_review_packet,
)
from hpo_mcp_server import validate_search_space


class TestLeanRoadmap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.study_name = "test_lean_" + self._testMethodName
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=os.environ["HPO_DATABASE_URL"],
            directions=["minimize", "maximize"],
            load_if_exists=True,
        )

    def _complete_trial(self, loss: float, score: float):
        self.study.enqueue_trial({"learning_rate": 1e-3, "batch_size": 8})
        t = self.study.ask()
        self.study.tell(t.number, [loss, score])

    def test_statistical_confidence_tiers(self):
        self.assertEqual(compute_statistical_confidence(5), "low")
        self.assertEqual(compute_statistical_confidence(15), "medium")
        self.assertEqual(compute_statistical_confidence(25), "high")
        packet = build_review_packet(self.study_name)
        self.assertIn("statistical_confidence", packet)

    def test_validate_review_fields_contract(self):
        bad = validate_review_fields(None, None)
        self.assertFalse(bad["ok"])
        good = validate_review_fields(0.03, 0)
        self.assertTrue(good["ok"])

    def test_coordinator_accuracy_insufficient_data(self):
        acc = compute_coordinator_accuracy(self.study_name)
        self.assertTrue(acc["insufficient_data"])
        self.assertEqual(acc["n_scored_reviews"], 0)

    def test_backfill_measured_outcome(self):
        for i in range(6):
            self._complete_trial(0.5 - i * 0.01, 0.5 + i * 0.02)

        save_study_review(
            self.study_name,
            "Narrow LR for gain",
            health_rating=4,
            policy_action="update_active_search_space",
            trials_evaluated=6,
            estimated_dice_improvement=0.05,
            cited_best_trial=5,
            force=True,
        )
        mark_review_applied(self.study_name)

        for i in range(5):
            self._complete_trial(0.3, 0.7 + i * 0.01)

        backfill_review_outcomes(self.study_name)
        with get_db_session() as session:
            review = (
                session.query(StudyReview)
                .filter_by(study_name=self.study_name)
                .order_by(StudyReview.id.desc())
                .first()
            )
            self.assertEqual(review.outcome_status, "measured")
            self.assertIsNotNone(review.actual_score_improvement)

        acc = compute_coordinator_accuracy(self.study_name)
        self.assertEqual(acc["n_scored_reviews"], 1)
        self.assertTrue(acc["insufficient_data"])

    def test_validate_search_space_empty_tunable(self):
        pinned = {
            "learning_rate": {"min": 1e-3, "max": 1e-3, "type": "float_log"},
            "batch_size": {"options": [8], "active": [8], "type": "categorical"},
        }
        result = validate_search_space(pinned)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
