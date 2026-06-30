"""Study health monitoring — health tier computation, statistical confidence, status file.

This module replaces the health-monitoring portions of the deleted hpo_coordinator.py.
It contains no LLM-calling code, no review persistence, no coordinator logic. It is a
deterministic, read-only health assessment layer imported by the broker, daemon, and CLI.
"""

import datetime
import json
import math
from typing import Dict, List, Optional, Tuple

from optuna.trial import TrialState

from .db_manager import get_db_session
from .metrics import get_score, get_completed_trials, TERMINAL_STATES

_MIN_COMPLETED_FOR_FIRST_REVIEW = 5


def compute_statistical_confidence(n_complete: int) -> str:
    if n_complete < 10:
        return "low"
    if n_complete < 20:
        return "medium"
    return "high"


def count_evaluated_trials(study) -> int:
    return sum(1 for t in study.trials if t.state in TERMINAL_STATES)


def compute_health_tier(study, study_name: str) -> Tuple[str, Optional[str]]:

    trials = list(study.trials)
    finished = sorted([t for t in trials if t.state in TERMINAL_STATES], key=lambda t: t.number)
    completed = sorted(get_completed_trials(study), key=lambda t: t.number)

    # === Intervene triggers ===

    for t in trials:
        if t.values:
            for v in t.values:
                if v is not None and (math.isnan(v) or math.isinf(v)):
                    return "intervene", f"NaN or Inf detected in reported metrics for Trial #{t.number}"

    try:
        from .schema import TrialResult
        with get_db_session() as session:
            oom_trials = session.query(TrialResult).filter_by(study_name=study_name, oom_triggered=True).all()
            if len(oom_trials) >= 2:
                trial_params_map = {t._trial_id: t.params for t in study.trials}
                oom_combos: Dict[tuple, int] = {}
                for r in oom_trials:
                    params = trial_params_map.get(r.trial_id)
                    if params:
                        key = tuple(sorted((k, str(v)) for k, v in params.items()))
                        oom_combos[key] = oom_combos.get(key, 0) + 1
                        if oom_combos[key] >= 2:
                            params_desc = ", ".join(f"{k}={v}" for k, v in key)
                            return "intervene", f"OOM cluster detected: parameter combination ({params_desc}) failed with OOM 2+ times"
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error checking OOM clusters: {e}")

    if len(completed) >= 5:
        improvements: List[int] = []
        best_so_far = -float("inf")
        for i, t in enumerate(completed):
            score = get_score(t, study)
            if score is None:
                continue
            if score > best_so_far + 1e-4:
                best_so_far = score
                improvements.append(i)
        if len(improvements) >= 2:
            intervals = [improvements[j] - improvements[j - 1] for j in range(1, len(improvements))]
            avg_interval = sum(intervals) / len(intervals)
            trials_since = len(completed) - 1 - improvements[-1]
            threshold = max(4, int(math.ceil(2 * avg_interval)))
            if trials_since >= threshold:
                return "intervene", f"Score stagnation: no improvement over last {trials_since} completed trials (average improvement interval is {avg_interval:.1f} trials, threshold is {threshold})"

    if len(completed) >= 4:
        from .hpo_config import load_hpo_config
        config = load_hpo_config(study_name)
        score_fixed_key = config.get("eval_protocol", {}).get("fixed_score_attr", "score_eval_fixed")
        gaps: List[float] = []
        for t in completed:
            fd = t.user_attrs.get(score_fixed_key)
            td = get_score(t, study)
            if fd is not None and td is not None:
                gaps.append(float(td) - float(fd))
        if len(gaps) >= 4:
            mean_gap = sum(gaps) / len(gaps)
            var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            std_gap = var_gap ** 0.5
            if std_gap > 0 and gaps[-1] > mean_gap + 2 * std_gap:
                return "intervene", f"Train-eval gap anomaly: latest trial gap ({gaps[-1]:.4f}) exceeds 2 standard deviations of historical gap distribution (mean={mean_gap:.4f}, std={std_gap:.4f}, threshold={mean_gap + 2 * std_gap:.4f})"

    # === Watch triggers ===

    if len(finished) >= 5:
        recent_finished = finished[-5:]
        pruned_count = sum(1 for t in recent_finished if t.state == TrialState.PRUNED)
        if pruned_count >= 4:
            return "watch", f"High prune rate: {pruned_count}/5 ({pruned_count * 20}%) of recent trials were pruned"

    if len(completed) >= 4:
        scores = [get_score(t, study) for t in completed]
        scores = [s for s in scores if s is not None]
        scores.sort(reverse=True)
        top_count = max(1, len(scores) // 4)
        top_scores = scores[:top_count]
        if len(top_scores) >= 2:
            mean_top = sum(top_scores) / len(top_scores)
            var_top = sum((x - mean_top) ** 2 for x in top_scores) / len(top_scores)
            if var_top < 1e-4:
                return "watch", f"Score convergence: top quartile score variance ({var_top:.6f}) is below 1e-4"

    running_trials = [t for t in trials if t.state == TrialState.RUNNING]
    latest_completed = completed[-1:] if completed else []
    for t in (running_trials + latest_completed):
        t_tier = t.user_attrs.get("health_tier")
        t_reason = t.user_attrs.get("health_reason")
        if t_tier == "watch" and t_reason:
            return "watch", f"Trial #{t.number} warning: {t_reason}"

    return "healthy", "No issues detected. Search space is healthy."


def write_ide_status_file(study_name: str, health_tier: str, health_reason: str, study) -> None:
    trials_evaluated = count_evaluated_trials(study)
    payload = {
        "study_name": study_name,
        "health_tier": health_tier.lower(),
        "health_reason": health_reason,
        "trials_evaluated": trials_evaluated,
        "review_recommended": health_tier.lower() in ("watch", "intervene"),
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    status_file_path = root_dir / ".hpo_status.json"
    try:
        with open(status_file_path, "w") as f:
            json.dump(payload, f, indent=4)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error writing .hpo_status.json: {e}")
