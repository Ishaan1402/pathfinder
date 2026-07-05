"""Generic metric helpers — replace hardcoded t.values[0]/t.values[1] assumptions.

Every function in this module derives objective indices from the Optuna study's
``directions`` list instead of assuming [minimize, maximize] order.  This makes the
codebase work for:
  - single-objective (maximize *or* minimize)
  - multi-objective with up to 2 objectives (one maximize, one minimize)
"""

from typing import List, Optional, Sequence
import math
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

TERMINAL_STATES = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)

# ---------------------------------------------------------------------------
# Objective-index discovery (look at directions, not hard-coded positions)
# ---------------------------------------------------------------------------

def _directions(study) -> Sequence[StudyDirection]:
    """Return the study's directions as a sequence.

    Works for both multi-objective (list) and single-objective studies.
    """
    # optuna stores directions as a list even for single-objective studies
    if hasattr(study, "directions") and study.directions:
        return study.directions
    # ultra-defensive fallback
    return []


def score_objective_index(study) -> Optional[int]:
    """Index of the **first** MAXIMIZE direction, or ``None``."""
    for i, d in enumerate(_directions(study)):
        if d == StudyDirection.MAXIMIZE:
            return i
    return None


def loss_objective_index(study) -> Optional[int]:
    """Index of the **first** MINIMIZE direction, or ``None``."""
    for i, d in enumerate(_directions(study)):
        if d == StudyDirection.MINIMIZE:
            return i
    return None


# ---------------------------------------------------------------------------
# Value access (single trial)
# ---------------------------------------------------------------------------

def _raw_value(trial: FrozenTrial, index: int) -> Optional[float]:
    """Safely retrieve ``trial.values[index]``, falling back to ``trial.value`` for
    single-objective studies."""
    if trial.values and len(trial.values) > index:
        return trial.values[index]
    if index == 0 and trial.value is not None:
        return float(trial.value)
    return None


def get_score(trial: FrozenTrial, study) -> Optional[float]:
    """Primary *score* (first MAXIMIZE objective) for a single trial."""
    idx = score_objective_index(study)
    if idx is None:
        return None
    return _raw_value(trial, idx)


def get_loss(trial: FrozenTrial, study) -> Optional[float]:
    """Primary *loss* (first MINIMIZE objective) for a single trial."""
    idx = loss_objective_index(study)
    if idx is None:
        return None
    return _raw_value(trial, idx)


def get_best_trial(trials: List[FrozenTrial], study) -> Optional[FrozenTrial]:
    """Return the trial with the highest primary score (MAXIMIZE objective)."""
    best = None
    best_val = float("-inf")
    for t in trials:
        v = get_score(t, study)
        if v is not None and v > best_val:
            best_val = v
            best = t
    return best


def get_best_score(trials: List[FrozenTrial], study) -> Optional[float]:
    """Highest primary score among the given trials."""
    best = get_best_trial(trials, study)
    return get_score(best, study) if best else None


# ---------------------------------------------------------------------------
# Direction-list variants (for callers that have directions but not a study obj)
# ---------------------------------------------------------------------------

def score_objective_index_from_dirs(directions: Sequence[StudyDirection]) -> Optional[int]:
    """Same as ``score_objective_index`` but accepts a raw directions list."""
    for i, d in enumerate(directions):
        if d == StudyDirection.MAXIMIZE:
            return i
    return None


def loss_objective_index_from_dirs(directions: Sequence[StudyDirection]) -> Optional[int]:
    """Same as ``loss_objective_index`` but accepts a raw directions list."""
    for i, d in enumerate(directions):
        if d == StudyDirection.MINIMIZE:
            return i
    return None


def get_score_from_dirs(trial: FrozenTrial, directions: Sequence[StudyDirection]) -> Optional[float]:
    """Primary score for a trial when only the directions list is available."""
    idx = score_objective_index_from_dirs(directions)
    if idx is None:
        return None
    return _raw_value(trial, idx)


def get_loss_from_dirs(trial: FrozenTrial, directions: Sequence[StudyDirection]) -> Optional[float]:
    """Primary loss for a trial when only the directions list is available."""
    idx = loss_objective_index_from_dirs(directions)
    if idx is None:
        return None
    return _raw_value(trial, idx)


def _trial_metric_snapshot(
    trial: FrozenTrial,
    history: List[dict],
    score_fixed_attr: str,
    loss_fixed_attr: str,
    directions: Sequence[StudyDirection] = None,
) -> dict:
    """Score/Loss for dashboard: completed values, else latest epoch / user_attrs."""
    _score = _loss = _score_eval_fixed = _loss_eval_fixed = None
    latest_epoch = trial.user_attrs.get("latest_epoch")

    from optuna.trial import TrialState
    from optuna.study import StudyDirection

    if trial.state == TrialState.COMPLETE and (trial.values or trial.value is not None):
        if trial.values and len(trial.values) > 1:
            _loss = get_loss_from_dirs(trial, directions or [])
            _score = get_score_from_dirs(trial, directions or [])
        else:
            if directions and directions[0] == StudyDirection.MINIMIZE:
                _loss = trial.value
            else:
                _score = trial.value
    else:
        _score = trial.user_attrs.get("latest_score")
        _loss = trial.user_attrs.get("latest_loss")
        _score_eval_fixed = trial.user_attrs.get("score_eval_fixed", trial.user_attrs.get(score_fixed_attr))
        _loss_eval_fixed = trial.user_attrs.get("loss_eval_fixed", trial.user_attrs.get(loss_fixed_attr))

    if history:
        last = max(history, key=lambda e: e.get("epoch", 0))
        latest_epoch = latest_epoch or last.get("epoch")
        if _score is None:
            _score = last.get("score")
        if _loss is None:
            _loss = last.get("loss")
        if _score_eval_fixed is None:
            _score_eval_fixed = last.get("score_eval_fixed")
        if _loss_eval_fixed is None:
            _loss_eval_fixed = last.get("loss_eval_fixed")

    if _score_eval_fixed is None:
        _score_eval_fixed = trial.user_attrs.get("score_eval_fixed", trial.user_attrs.get(score_fixed_attr))
    if _loss_eval_fixed is None:
        _loss_eval_fixed = trial.user_attrs.get("loss_eval_fixed", trial.user_attrs.get(loss_fixed_attr))

    return {
        "score": _score,
        "loss": _loss,
        "score_eval_fixed": _score_eval_fixed,
        "loss_eval_fixed": _loss_eval_fixed,
        "latest_epoch": latest_epoch,
    }

def get_eval_attr_names(ev: dict) -> tuple[str, str]:
    """Return the score and loss user-attribute names for fixed-eval tracking."""
    score_fixed_key = ev.get("fixed_score_attr", "score_eval_fixed")
    loss_fixed_key = ev.get("fixed_loss_attr", "loss_eval_fixed")
    return score_fixed_key, loss_fixed_key

def get_completed_trials(study) -> List[FrozenTrial]:
    """Return all COMPLETE trials in the study."""
    return [t for t in study.trials if t.state == TrialState.COMPLETE]

def has_invalid_metrics(**kwargs) -> Optional[str]:
    """Check if any of the provided metrics are NaN or Inf. Returns the name of the first invalid metric."""
    for val_name, val in kwargs.items():
        if val is not None and (math.isnan(val) or math.isinf(val)):
            return val_name
    return None
