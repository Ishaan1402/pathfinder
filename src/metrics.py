"""Generic metric helpers — replace hardcoded t.values[0]/t.values[1] assumptions.

Every function in this module derives objective indices from the Optuna study's
``directions`` list instead of assuming [minimize, maximize] order.  This makes the
codebase work for:
  - single-objective (maximize *or* minimize)
  - multi-objective with any number of objectives and any direction ordering
  - the original 2-obj [minimize, maximize] setup (fully backward-compatible)
"""

from typing import Any, List, Optional, Sequence
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


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
    dice_fixed_attr: str,
    bce_fixed_attr: str,
    directions: Sequence[StudyDirection] = None,
) -> dict:
    """Score/Loss for dashboard: completed values, else latest epoch / user_attrs."""
    bce = dice = dice_eval_fixed = bce_eval_fixed = None
    latest_epoch = trial.user_attrs.get("latest_epoch")

    from optuna.trial import TrialState
    from optuna.study import StudyDirection

    if trial.state == TrialState.COMPLETE and (trial.values or trial.value is not None):
        if trial.values and len(trial.values) > 1:
            bce = get_loss_from_dirs(trial, directions or [])
            dice = get_score_from_dirs(trial, directions or [])
        else:
            if directions and directions[0] == StudyDirection.MINIMIZE:
                bce = trial.value
            else:
                dice = trial.value
    else:
        dice = trial.user_attrs.get("latest_dice")
        bce = trial.user_attrs.get("latest_bce")
        dice_eval_fixed = trial.user_attrs.get(dice_fixed_attr)
        bce_eval_fixed = trial.user_attrs.get(bce_fixed_attr)

    if history:
        last = max(history, key=lambda e: e.get("epoch", 0))
        latest_epoch = latest_epoch or last.get("epoch")
        if dice is None:
            dice = last.get("dice")
        if bce is None:
            bce = last.get("bce")
        if dice_eval_fixed is None:
            dice_eval_fixed = last.get("dice_eval_fixed")
        if bce_eval_fixed is None:
            bce_eval_fixed = last.get("bce_eval_fixed")

    if dice_eval_fixed is None:
        dice_eval_fixed = trial.user_attrs.get(dice_fixed_attr)
    if bce_eval_fixed is None:
        bce_eval_fixed = trial.user_attrs.get(bce_fixed_attr)

    return {
        "bce": bce,
        "dice": dice,
        "dice_eval_fixed": dice_eval_fixed,
        "bce_eval_fixed": bce_eval_fixed,
        "latest_epoch": latest_epoch,
    }
