# ruff: noqa: E402
import os
import sys
import unittest
from unittest.mock import patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import optuna
from optuna.trial import TrialState

from src.metrics import (
    get_best_score,
    get_best_trial,
    get_loss,
    get_loss_from_dirs,
    get_score,
    get_score_from_dirs,
    loss_objective_index,
    score_objective_index,
)
from src.analytics import get_fanova_importances


def _complete_trial(study, values, params=None):
    study.enqueue_trial(params or {"x": 0.5})
    trial = study.ask()
    if isinstance(values, (int, float)):
        study.tell(trial.number, values)
    else:
        study.tell(trial.number, list(values))
    return study.trials[-1]


class TestMetrics(unittest.TestCase):
    def test_legacy_minimize_maximize_indices(self):
        study = optuna.create_study(directions=["minimize", "maximize"])
        self.assertEqual(loss_objective_index(study), 0)
        self.assertEqual(score_objective_index(study), 1)

    def test_reversed_maximize_minimize_indices(self):
        study = optuna.create_study(directions=["maximize", "minimize"])
        self.assertEqual(score_objective_index(study), 0)
        self.assertEqual(loss_objective_index(study), 1)

    def test_single_objective_maximize(self):
        study = optuna.create_study(direction="maximize")
        trial = _complete_trial(study, 0.75)
        self.assertEqual(get_score(trial, study), 0.75)
        self.assertIsNone(get_loss(trial, study))

    def test_single_objective_minimize(self):
        study = optuna.create_study(direction="minimize")
        trial = _complete_trial(study, 0.42)
        self.assertIsNone(get_score(trial, study))
        self.assertEqual(get_loss(trial, study), 0.42)

    def test_get_best_trial_and_score(self):
        study = optuna.create_study(directions=["minimize", "maximize"])
        t1 = _complete_trial(study, [0.3, 0.9])
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        best = get_best_trial(completed, study)
        self.assertEqual(best.number, t1.number)
        self.assertEqual(get_best_score(completed, study), 0.9)

    def test_get_score_from_dirs(self):
        study = optuna.create_study(directions=["maximize", "minimize"])
        trial = _complete_trial(study, [0.8, 0.2])
        self.assertEqual(get_score_from_dirs(trial, study.directions), 0.8)
        self.assertEqual(get_loss_from_dirs(trial, study.directions), 0.2)

    def test_fanova_importances_no_score_objective(self):
        study = optuna.create_study(
            study_name="test_fanova_no_score_" + self._testMethodName,
            storage="sqlite:///:memory:",
            directions=["minimize", "minimize"],
            load_if_exists=True,
        )
        _complete_trial(study, [0.5, 0.3], {"x": 0.1})
        _complete_trial(study, [0.4, 0.2], {"x": 0.9})
        result = get_fanova_importances(study, {})
        self.assertEqual(result, {})

    def test_fanova_importances_mocked_score_index_none(self):
        study = optuna.create_study(directions=["minimize", "maximize"])
        _complete_trial(study, [0.5, 0.6])
        _complete_trial(study, [0.4, 0.7])
        with patch("src.analytics.score_objective_index", return_value=None):
            result = get_fanova_importances(study, {})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
