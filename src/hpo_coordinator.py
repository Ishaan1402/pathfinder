"""Shared coordinator logic for Pathfinder.

This module is the single source of truth for:
  - read-only review packets (Pareto, fANOVA, eval insights, recent trials),
  - deterministic "review recommended" heuristics (no LLM, no network),
  - study review persistence (idempotent across IDE clients).

It is imported by both broker.py (HTTP runtime) and hpo_mcp_server.py (MCP tools) so
the two surfaces speak the same language without duplicating the suggest path or the
drift-detection logic.
"""
from typing import Any, Dict, List, Optional

import datetime
import optuna
from optuna.trial import TrialState

import os
import json
from .db_manager import get_db_session, DATABASE_URL
from .schema import StudyReview, StudyStatus, TrialResult, SystemConfiguration, CompactedPacket
from .hpo_config import load_hpo_config, normalize_trial_params, param_display_name

# --- Tunables for drift heuristics (deterministic, no model calls) ---
RECENT_TRIALS_IN_PACKET = 8
PRUNE_STREAK_THRESHOLD = 3
STAGNATION_WINDOW = 5
MIN_COMPLETED_FOR_FIRST_REVIEW = 5

POLICY_ACTIONS = ("no_change", "update_active_search_space", "enqueue_one_manual_trial")

REVIEW_PROMPT = """You are the Pathfinder coordinator reviewing study '{study_name}'. Do NOT block the training worker.
1. Call get_study_data('{study_name}') to obtain the compacted review packet, which includes active search space constraints, HPO config, Spearman correlations, fANOVA, VRAM/OOM telemetry, boundary hits, past reviews, and prediction accuracy.
2. Analyze the packet data under the following constraints:
   - COORDINATOR DECISION MEMORY: Review `past_reviews` (up to the last 3 reviews) to maintain logical consistency. Do NOT blindly reverse previous decisions or flip-flop unless new evidence warrants it. However, do NOT copy previous decisions; critically analyze fresh trials and build on previous hypotheses.
   - SPEARMAN CORRELATIONS: Check the `spearman_correlations` confidence tags. Treat correlations with caution if the confidence is "Low" or "Moderate" (due to statistical noise at low sample sizes).
   - VRAM & OOM FORECASTS: Examine `bounds_oom_risk` under `vram_telemetry`. Do NOT treat mean predictions as facts. Treat `predicted_mean_vram_gb + margin_gb` (the predicted max VRAM) as the safety boundary relative to `gpu_capacity_gb`. Shrink/cap search bounds if there is high OOM risk.
   - ACCURACY SELF-REGULATION: Check `coordinator_accuracy` in the packet. If `insufficient_data` is true (fewer than 3 scored reviews), do not self-regulate yet. If `mean_absolute_error` > 0.05 with n_scored_reviews >= 3, be more conservative—propose smaller search space shifts. Ignore reviews where `quality_flagged` is true.
   - DYNAMIC METRIC LABELS: Refer to scores and losses using the dynamic labels specified in the packet (e.g. '{metric_score_label}' and '{metric_loss_label}').
   - GLOBAL BEST TRIAL: The current global best trial is: {best_trial_info}.
3. Rate search space health 1-5 (preferring fixed-eval score if available).
4. Provide a numeric forecast for estimated score improvement. If you have thin/insufficient data (e.g. fewer than 5 trials completed), use `-1.0` as a sentinel value.
5. Identify the trial number you cite as the best trial so far and provide it as the `cited_best_trial` parameter.
6. Select exactly ONE policy action: no_change, update_active_search_space (via update_search_space), or enqueue_one_manual_trial.
7. If proposing active search space changes, call the tool update_search_space(study_name, space_config, apply=False). If enqueuing a manual trial, pass the parameter dictionary as the `manual_trial` argument when calling submit_agent_review.
8. Call submit_agent_review. Write a 3-5 line summary focusing on specific trial numbers and stats, and including a Git-like diff of bounds changes if updated.
"""


def compute_statistical_confidence(n_complete: int) -> str:
    """Tiered confidence from completed-trial count (caveat only, never a hard gate)."""
    if n_complete < 10:
        return "low"
    if n_complete < 20:
        return "medium"
    return "high"


def get_best_primary_score(study) -> Optional[float]:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return None
    if len(study.directions) > 1:
        scores = [t.values[1] for t in completed if t.values and len(t.values) > 1]
        return max(scores) if scores else None
    return study.best_value


def validate_review_fields(
    estimated_score_improvement: Optional[float],
    cited_best_trial: Optional[int],
) -> Dict[str, Any]:
    """Required JSON contract for coordinator review submissions."""
    errors: List[str] = []
    if cited_best_trial is None:
        errors.append("cited_best_trial is required (int trial number).")
    if estimated_score_improvement is None:
        errors.append("estimated_score_improvement is required (float).")
    else:
        try:
            float(estimated_score_improvement)
        except (TypeError, ValueError):
            errors.append("estimated_score_improvement must be a number.")
    return {"ok": len(errors) == 0, "errors": errors}


def compute_coordinator_accuracy(study_name: str) -> Dict[str, Any]:
    """MAE from measured StudyReview outcomes (excludes inconclusive, sentinel, and flagged)."""
    with get_db_session() as session:
        rows = (
            session.query(StudyReview)
            .filter(
                StudyReview.study_name == study_name,
                StudyReview.outcome_status == "measured",
                StudyReview.quality_flagged.is_(False),
                StudyReview.estimated_score_improvement.isnot(None),
                StudyReview.actual_score_improvement.isnot(None),
            )
            .order_by(StudyReview.id.asc())
            .all()
        )
        scored = []
        for r in rows:
            if r.estimated_score_improvement == -1.0:
                continue
            scored.append({
                "review_id": r.id,
                "estimated_score_improvement": r.estimated_score_improvement,
                "actual_score_improvement": r.actual_score_improvement,
                "absolute_error": abs(r.estimated_score_improvement - r.actual_score_improvement),
            })
        n = len(scored)
        result: Dict[str, Any] = {
            "n_scored_reviews": n,
            "insufficient_data": n < 3,
            "mean_absolute_error": None,
            "accuracy_rate_05": None,
            "recent_predictions": scored[-5:],
        }
        if n > 0:
            errors = [s["absolute_error"] for s in scored]
            result["mean_absolute_error"] = sum(errors) / n
            result["accuracy_rate_05"] = sum(1 for e in errors if e <= 0.05) / n
        return result


def mark_review_applied(study_name: str) -> None:
    """Record when a coordinator search-space patch was committed."""
    study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    complete_count = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    now = datetime.datetime.utcnow()
    with get_db_session() as session:
        review = (
            session.query(StudyReview)
            .filter_by(study_name=study_name)
            .filter(StudyReview.policy_action != "no_change")
            .filter(StudyReview.outcome_status == "pending")
            .filter(StudyReview.applied_at_completed_count.is_(None))
            .order_by(StudyReview.created_at.desc(), StudyReview.id.desc())
            .first()
        )
        if review:
            review.applied_at_completed_count = complete_count
            review.applied_at = now


def backfill_review_outcomes(study_name: str) -> None:
    """Measure coordinator forecast accuracy after post-apply trial windows."""
    study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    complete_count = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    terminal_states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)

    with get_db_session() as session:
        pending = (
            session.query(StudyReview)
            .filter(
                StudyReview.study_name == study_name,
                StudyReview.outcome_status == "pending",
                StudyReview.policy_action != "no_change",
                StudyReview.applied_at_completed_count.isnot(None),
            )
            .all()
        )
        for review in pending:
            if review.applied_at is None:
                continue
            complete_since = complete_count - review.applied_at_completed_count
            applied_at = review.applied_at.replace(tzinfo=None) if review.applied_at.tzinfo else review.applied_at
            finished_since = [
                t for t in study.trials
                if t.state in terminal_states
                and t.datetime_complete
                and (
                    t.datetime_complete.replace(tzinfo=None)
                    if getattr(t.datetime_complete, "tzinfo", None)
                    else t.datetime_complete
                ) >= applied_at
            ]
            if len(finished_since) >= 15 and complete_since < 3:
                review.outcome_status = "inconclusive"
                review.outcome_measured_at = datetime.datetime.utcnow()
                continue
            if complete_since >= 5:
                new_best = get_best_primary_score(study)
                baseline = review.baseline_best_score
                if new_best is not None and baseline is not None:
                    review.actual_score_improvement = new_best - baseline
                    review.outcome_status = "measured"
                    review.outcome_measured_at = datetime.datetime.utcnow()


def flag_study_review(review_id: int, flagged: bool = True) -> Dict[str, Any]:
    with get_db_session() as session:
        review = session.query(StudyReview).filter_by(id=review_id).first()
        if not review:
            return {"success": False, "error": f"Review id {review_id} not found."}
        review.quality_flagged = flagged
        session.flush()
        return {"success": True, "review": review.to_dict()}


def build_review_prompt(study_name: str) -> str:
    config = load_hpo_config(study_name)
    score_label = config.get("metric_score_label", "Score")
    loss_label = config.get("metric_loss_label", "Loss")
    
    best_trial_info = "None (No trials completed yet)"
    stat_confidence = "low"
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        stat_confidence = compute_statistical_confidence(len(completed))
        if completed:
            if len(study.directions) > 1:
                best_t = max(completed, key=lambda t: t.values[1] if (t.values and len(t.values) > 1) else -float('inf'))
                score_val = best_t.values[1] if (best_t.values and len(best_t.values) > 1) else 0.0
            else:
                best_t = study.best_trial
                score_val = best_t.value
            best_trial_info = f"Trial #{best_t.number} with {score_label}: {score_val:.4f}"
    except Exception as e:
        best_trial_info = f"Error reading best trial: {e}"
        
    prompt = REVIEW_PROMPT.format(
        study_name=study_name,
        metric_score_label=score_label,
        metric_loss_label=loss_label,
        best_trial_info=best_trial_info
    )
    if stat_confidence == "low":
        prompt = (
            "STATISTICAL CONFIDENCE: LOW — fewer than 10 completed trials. "
            "Treat fANOVA and Spearman signals as noisy. Use estimated_score_improvement=-1.0 "
            "when you cannot justify a numeric forecast.\n\n"
        ) + prompt
    elif stat_confidence == "medium":
        prompt = (
            "STATISTICAL CONFIDENCE: MEDIUM — 10–19 completed trials. "
            "Correlations may stabilize but remain cautious on bound changes.\n\n"
        ) + prompt
    return prompt


def load_active_search_space(study_name: Optional[str] = None) -> Dict[str, Any]:
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
    with get_db_session() as session:
        row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        if row:
            try:
                return json.loads(row.config_value)
            except Exception:
                pass
    return {}


def get_ranks(v: List[float]) -> List[float]:
    n = len(v)
    indexed = sorted(enumerate(v), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def compute_spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    rx = get_ranks(x)
    ry = get_ranks(y)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return 0.0
    raw_corr = num / (den_x * den_y) ** 0.5
    return raw_corr


def compute_prune_rate_clusters(study, search_space: Dict[str, Any]) -> Dict[str, Any]:
    clusters = {}
    continuous_params = []
    for p_name, p_info in search_space.items():
        p_type = p_info.get("type", "")
        if p_type in ("float", "float_log", "int"):
            continuous_params.append(p_name)
    
    for p_name in continuous_params:
        vals = []
        states = []
        for t in study.trials:
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED) and p_name in t.params:
                val = t.params[p_name]
                if val is not None:
                    vals.append(float(val))
                    states.append(t.state)
        
        if not vals:
            continue
        
        v_min, v_max = min(vals), max(vals)
        if v_max <= v_min:
            continue
        
        w = (v_max - v_min) / 3.0
        bins = [
            {"min": v_min, "max": v_min + w, "total": 0, "pruned": 0},
            {"min": v_min + w, "max": v_min + 2*w, "total": 0, "pruned": 0},
            {"min": v_min + 2*w, "max": v_max, "total": 0, "pruned": 0}
        ]
        
        for val, state in zip(vals, states):
            if val <= bins[0]["max"]:
                bin_idx = 0
            elif val <= bins[1]["max"]:
                bin_idx = 1
            else:
                bin_idx = 2
            
            bins[bin_idx]["total"] += 1
            if state == TrialState.PRUNED:
                bins[bin_idx]["pruned"] += 1
        
        for b in bins:
            b["prune_rate"] = b["pruned"] / b["total"] if b["total"] > 0 else 0.0
        
        clusters[p_name] = bins
    return clusters


def check_boundary_hits(study, pareto_numbers: List[int], search_space: Dict[str, Any]) -> Dict[str, Any]:
    hits = {}
    pareto_trials = [t for t in study.trials if t.number in pareto_numbers and t.state == TrialState.COMPLETE]
    n_pareto = len(pareto_trials)
    if n_pareto == 0:
        return hits
    
    for p_name, p_info in search_space.items():
        p_type = p_info.get("type", "")
        if p_type not in ("float", "float_log", "int"):
            continue
        
        s_min = p_info.get("min")
        s_max = p_info.get("max")
        if s_min is None or s_max is None or s_max <= s_min:
            continue
            
        s_min, s_max = float(s_min), float(s_max)
        margin = 0.1 * (s_max - s_min)
        
        near_min_count = 0
        near_max_count = 0
        
        for t in pareto_trials:
            val = t.params.get(p_name)
            if val is not None:
                val = float(val)
                if val <= s_min + margin:
                    near_min_count += 1
                if val >= s_max - margin:
                    near_max_count += 1
        
        total_hits = near_min_count + near_max_count
        ratio = total_hits / n_pareto
        
        if ratio > 0.6:
            hits[p_name] = {
                "near_min_count": near_min_count,
                "near_max_count": near_max_count,
                "total_pareto": n_pareto,
                "hit_ratio": ratio,
                "bound_hit": "min" if near_min_count > near_max_count else "max" if near_max_count > near_min_count else "both"
            }
    return hits


def compute_fidelity_durations(study, config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    
    groups = {}
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        val = t.params.get(train_param)
        if val is None:
            continue
        try:
            val = int(val)
        except (TypeError, ValueError):
            continue
        
        groups.setdefault(val, []).append(t)
        
    res_stats = {}
    for val, trials in groups.items():
        durations = []
        epoch_durations = []
        for t in trials:
            if t.datetime_start and t.datetime_complete:
                dur = (t.datetime_complete - t.datetime_start).total_seconds()
                durations.append(dur)
                history = t.user_attrs.get("history", [])
                epochs = len(history) if history else t.user_attrs.get("latest_epoch")
                if not epochs:
                    epochs = max([h.get("epoch", 1) for h in history] or [1])
                if epochs > 0:
                    epoch_durations.append(dur / epochs)
        
        if durations:
            res_stats[val] = {
                "avg_total_duration": sum(durations) / len(durations),
                "avg_epoch_duration": sum(epoch_durations) / len(epoch_durations) if epoch_durations else None,
                "count": len(durations)
            }
    
    if not res_stats:
        return {}
        
    lowest_scale = min(res_stats.keys())
    base_dur = res_stats[lowest_scale]["avg_total_duration"]
    
    for val, stats in res_stats.items():
        if base_dur > 0:
            stats["overhead_ratio"] = stats["avg_total_duration"] / base_dur
        else:
            stats["overhead_ratio"] = 1.0
            
    return {
        "fidelity_param": train_param,
        "lowest_scale": lowest_scale,
        "scales": res_stats
    }
def fit_vram_model(trials: List[Any], db_metrics: Dict[int, Any], train_param: str) -> Optional[Dict[str, Any]]:
    points = []
    for t in trials:
        if t.state != TrialState.COMPLETE:
            continue
        metric = db_metrics.get(t.number, {})
        oom = metric.get("oom_triggered") or t.user_attrs.get("oom_triggered", False)
        if oom:
            continue
        vram = metric.get("max_vram_gb") or t.user_attrs.get("max_vram_gb")
        if vram is None:
            continue
        bs = t.params.get("batch_size")
        res = t.params.get(train_param)
        if bs is not None and res is not None:
            try:
                points.append((float(bs), float(res), float(vram)))
            except (ValueError, TypeError):
                continue

    # We require at least 6 points to fit the model to prevent overfitting
    if len(points) < 6:
        return None
        
    X = [p[0] * (p[1] ** 2) for p in points]
    Y = [p[2] for p in points]
    
    # Verify variance in X
    if len(set(X)) < 2:
        return None
        
    n = len(points)
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_xx = sum(x*x for x in X)
    sum_xy = sum(X[i]*Y[i] for i in range(n))
    
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None
        
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate Residual Standard Error
    ssr = sum((Y[i] - (slope * X[i] + intercept)) ** 2 for i in range(n))
    rse = (ssr / (n - 2)) ** 0.5 if n > 2 else 0.0
    
    return {
        "slope": slope,
        "intercept": intercept,
        "n_points": n,
        "rse": rse
    }


def compute_vram_telemetry(study, db_metrics: Dict[int, Any], search_space: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    
    trials = list(study.trials)
    model = fit_vram_model(trials, db_metrics, train_param)
    
    gpu_capacity_gb = 0.0
    gpu_models = []
    oom_count = 0
    
    for t in trials:
        metric = db_metrics.get(t.number, {})
        vram = metric.get("max_vram_gb") or t.user_attrs.get("max_vram_gb")
        gpu = metric.get("gpu_model") or t.user_attrs.get("gpu_model")
        oom = metric.get("oom_triggered") or t.user_attrs.get("oom_triggered", False)
        
        if vram:
            gpu_capacity_gb = max(gpu_capacity_gb, float(vram))
        if gpu:
            gpu_models.append(gpu)
        if oom:
            oom_count += 1
            
    gpu_model = max(set(gpu_models), key=gpu_models.count) if gpu_models else "Unknown"
    
    oom_risk = None
    if model and gpu_capacity_gb > 0:
        max_bs = None
        bs_info = search_space.get("batch_size", {})
        if bs_info.get("type") == "categorical":
            active_bs = bs_info.get("active", [])
            if active_bs:
                max_bs = max(active_bs)
        else:
            max_bs = bs_info.get("max")
            
        max_res = None
        res_info = search_space.get(train_param, {})
        if res_info.get("type") == "categorical":
            active_res = res_info.get("active", [])
            if active_res:
                max_res = max(active_res)
        else:
            max_res = res_info.get("max")
            
        if max_bs is not None and max_res is not None:
            predicted_mean_vram = model["slope"] * (float(max_bs) * (float(max_res) ** 2)) + model["intercept"]
            margin = max(1.0, 1.96 * model["rse"])
            predicted_max_vram = predicted_mean_vram + margin
            
            if predicted_max_vram > 0.9 * gpu_capacity_gb:
                oom_risk = {
                    "max_batch_size": max_bs,
                    "max_resolution": max_res,
                    "predicted_mean_vram_gb": predicted_mean_vram,
                    "margin_gb": margin,
                    "predicted_max_vram_gb": predicted_max_vram,
                    "gpu_capacity_gb": gpu_capacity_gb,
                    "risk_level": "high" if predicted_max_vram > gpu_capacity_gb else "medium"
                }
                
    return {
        "gpu_model": gpu_model,
        "gpu_capacity_gb": gpu_capacity_gb,
        "oom_count": oom_count,
        "vram_model": model,
        "bounds_oom_risk": oom_risk
    }


# --- Train-resolution helper (shared with broker pruning/pareto) ---
def trial_train_resolution(trial, train_param: str) -> Optional[int]:
    val = trial.params.get(train_param)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def pareto_trial_numbers_deploy_aware(study, hpo_config: Dict[str, Any]) -> List[int]:
    """
    Pareto set for dashboard: optional filter excluding low train-res trials that inflate Dice.
    Uses fixed-eval Dice when available on completed trials.
    """
    ev = hpo_config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    low_warn = ev.get("low_train_res_warning")
    low_warn = int(low_warn) if low_warn is not None else None
    deploy_only = ev.get("pareto_deploy_resolution_only", True)
    dice_fixed_key = ev.get("fixed_dice_attr", "dice_eval_fixed")

    points: List[tuple] = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE or not t.values or len(t.values) < 2:
            continue
        train_res = trial_train_resolution(t, train_param)
        if deploy_only and low_warn is not None and train_res is not None and train_res < low_warn:
            continue
        bce = float(t.values[0])
        dice = float(t.values[1])
        if ev.get("enabled"):
            fd = t.user_attrs.get(dice_fixed_key)
            if fd is not None:
                dice = float(fd)
        points.append((t.number, bce, dice))

    if not points:
        try:
            return [t.number for t in study.best_trials]
        except Exception:
            return []

    pareto: List[int] = []
    for num_i, bce_i, dice_i in points:
        dominated = False
        for num_j, bce_j, dice_j in points:
            if num_i == num_j:
                continue
            if bce_j <= bce_i and dice_j >= dice_i and (bce_j < bce_i or dice_j > dice_i):
                dominated = True
                break
        if not dominated:
            pareto.append(num_i)
    return pareto


def study_eval_insights(study, config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    fixed_res = ev.get("fixed_resolution")
    low_warn = ev.get("low_train_res_warning")
    dice_fixed_key = ev.get("fixed_dice_attr", "dice_eval_fixed")

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    by_res: Dict[int, List] = {}
    warnings = []

    for t in complete:
        tr = t.params.get(train_param)
        if tr is not None:
            by_res.setdefault(int(tr), []).append(t)

    res_summary = {}
    for res, trials in sorted(by_res.items()):
        dices = [t.values[1] for t in trials if t.values and len(t.values) > 1]
        fixed_dices = [
            t.user_attrs.get(dice_fixed_key)
            for t in trials
            if t.user_attrs.get(dice_fixed_key) is not None
        ]
        res_summary[res] = {
            "count": len(trials),
            "best_dice_train": max(dices) if dices else None,
            "best_dice_fixed": max(fixed_dices) if fixed_dices else None,
        }

    # Suppress low-fidelity warning if a valid deploy-scale candidate exists (res >= low_warn)
    valid_deploy_exists = False
    if low_warn is not None:
        for t in complete:
            tr = t.params.get(train_param)
            fd = t.user_attrs.get(dice_fixed_key)
            if tr is not None and int(tr) >= int(low_warn) and fd is not None:
                valid_deploy_exists = True
                break

    if complete and ev.get("enabled") and fixed_res:
        best_train = max(complete, key=lambda t: t.values[1] if t.values else -1)
        train_res = best_train.params.get(train_param)
        if train_res is not None and low_warn and int(train_res) < int(low_warn) and not valid_deploy_exists:
            warnings.append(
                {
                    "code": "low_train_res_pareto",
                    "trial_number": best_train.number,
                    "message": (
                        f"Pareto-best trial #{best_train.number} trained at scale {train_res}, "
                        f"below warning threshold {low_warn}. Check {ev.get('dice_fixed_label', 'fixed eval')}."
                    ),
                }
            )
        fd = best_train.user_attrs.get(dice_fixed_key)
        td = best_train.values[1] if best_train.values else None
        if fd is not None and td is not None and (td - fd) > 0.08:
            warnings.append(
                {
                    "code": "train_eval_gap",
                    "trial_number": best_train.number,
                    "message": (
                        f"Trial #{best_train.number}: train score {td:.3f} vs "
                        f"fixed-eval score {fd:.3f} — train scale/resolution may be inflating scores."
                    ),
                }
            )

    best_deploy = None
    if ev.get("enabled"):
        ranked = [
            t
            for t in complete
            if t.user_attrs.get(dice_fixed_key) is not None
        ]
        if ranked:
            best_deploy = max(ranked, key=lambda t: t.user_attrs.get(dice_fixed_key))

    return {
        "resolution_summary": res_summary,
        "warnings": warnings,
        "best_deploy_trial_number": best_deploy.number if best_deploy else None,
        "best_deploy_dice_fixed": (
            best_deploy.user_attrs.get(dice_fixed_key) if best_deploy else None
        ),
    }


def _deploy_or_train_dice(trial, dice_fixed_key: str) -> Optional[float]:
    """Fixed-eval Dice if present, else train Dice from study values."""
    fd = trial.user_attrs.get(dice_fixed_key)
    if fd is not None:
        return float(fd)
    if trial.values and len(trial.values) > 1:
        return float(trial.values[1])
    return None


def count_evaluated_trials(study) -> int:
    """Finished trials (COMPLETE / PRUNED / FAIL) — the idempotency window for reviews."""
    return sum(
        1
        for t in study.trials
        if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    )


def compute_health_tier(study, study_name: str) -> tuple[str, Optional[str]]:
    """Evaluates study health using a tiered severity model (Healthy, Watch, Intervene).
    
    Returns (health_tier, health_reason)
    """
    import math
    from optuna.trial import TrialState
    
    trials = list(study.trials)
    finished = sorted([t for t in trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)], key=lambda t: t.number)
    completed = sorted([t for t in trials if t.state == TrialState.COMPLETE], key=lambda t: t.number)
    
    # 🔴 Intervene Triggers
    
    # 1. NaN or Inf detected in reported metrics
    for t in trials:
        if t.values:
            for v in t.values:
                if v is not None and (math.isnan(v) or math.isinf(v)):
                    return "intervene", f"NaN or Inf detected in reported metrics for Trial #{t.number}"
                    
    # 2. Same parameter combination caused 2+ OOM failures
    try:
        with get_db_session() as session:
            oom_trials = session.query(TrialResult).filter_by(study_name=study_name, oom_triggered=True).all()
            if len(oom_trials) >= 2:
                trial_params_map = {t._trial_id: t.params for t in study.trials}
                oom_combos = {}
                for r in oom_trials:
                    params = trial_params_map.get(r.trial_id)
                    if params:
                        key = tuple(sorted((k, str(v)) for k, v in params.items()))
                        oom_combos[key] = oom_combos.get(key, 0) + 1
                        if oom_combos[key] >= 2:
                            params_desc = ", ".join(f"{k}={v}" for k, v in key)
                            return "intervene", f"OOM cluster detected: parameter combination ({params_desc}) failed with OOM 2+ times"
    except Exception as e:
        print(f"Error checking OOM clusters: {e}")

    # 3. Best score hasn't improved in 2x the study's average improvement interval
    if len(completed) >= 5:
        improvements = []
        best_so_far = -float("inf")
        for i, t in enumerate(completed):
            score = t.values[1] if len(t.values) > 1 else t.values[0]
            if score > best_so_far + 1e-4:
                best_so_far = score
                improvements.append(i)
                
        if len(improvements) >= 2:
            intervals = [improvements[j] - improvements[j-1] for j in range(1, len(improvements))]
            avg_interval = sum(intervals) / len(intervals)
            trials_since_last_improvement = len(completed) - 1 - improvements[-1]
            threshold = max(4, int(math.ceil(2 * avg_interval)))
            if trials_since_last_improvement >= threshold:
                return "intervene", f"Score stagnation: no improvement over last {trials_since_last_improvement} completed trials (average improvement interval is {avg_interval:.1f} trials, threshold is {threshold})"

    # 4. Train-eval metric gap exceeds 2σ of the study's historical gap distribution
    if len(completed) >= 4:
        config = load_hpo_config(study_name)
        dice_fixed_key = config.get("eval_protocol", {}).get("fixed_dice_attr", "dice_eval_fixed")
        gaps = []
        for t in completed:
            fd = t.user_attrs.get(dice_fixed_key)
            td = t.values[1] if len(t.values) > 1 else t.values[0]
            if fd is not None and td is not None:
                gaps.append(float(td) - float(fd))
                
        if len(gaps) >= 4:
            mean_gap = sum(gaps) / len(gaps)
            var_gap = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            std_gap = var_gap ** 0.5
            latest_gap = gaps[-1]
            if std_gap > 0 and latest_gap > mean_gap + 2 * std_gap:
                return "intervene", f"Train-eval gap anomaly: latest trial gap ({latest_gap:.4f}) exceeds 2 standard deviations of historical gap distribution (mean={mean_gap:.4f}, std={std_gap:.4f}, threshold={mean_gap + 2*std_gap:.4f})"

    # 🟡 Watch Triggers
    
    # 1. Prune rate over last 5 trials exceeds 80% (>= 4 out of last 5 finished trials are pruned)
    if len(finished) >= 5:
        recent_finished = finished[-5:]
        pruned_count = sum(1 for t in recent_finished if t.state == TrialState.PRUNED)
        if pruned_count >= 4:
            return "watch", f"High prune rate: {pruned_count}/5 ({pruned_count*20}%) of recent trials were pruned"
            
    # 2. Score variance in top quartile drops below epsilon (stagnation)
    if len(completed) >= 4:
        scores = []
        for t in completed:
            score = t.values[1] if len(t.values) > 1 else t.values[0]
            scores.append(score)
        scores.sort(reverse=True)
        top_quartile_count = max(1, len(scores) // 4)
        top_scores = scores[:top_quartile_count]
        if len(top_scores) >= 2:
            mean_top = sum(top_scores) / len(top_scores)
            var_top = sum((x - mean_top) ** 2 for x in top_scores) / len(top_scores)
            if var_top < 1e-4:
                return "watch", f"Score convergence: top quartile score variance ({var_top:.6f}) is below 1e-4"

    return "healthy", "No issues detected. Search space is healthy."


def compute_review_heuristics(
    study, insights: Dict[str, Any], config: Dict[str, Any], study_name: str
) -> Dict[str, Any]:
    """Simplified heuristics adapter using compute_health_tier."""
    health_tier, health_reason = compute_health_tier(study, study_name)
    review_recommended = health_tier in ("watch", "intervene")
    finished = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
    n_eval = len(finished)
    latest = get_latest_study_review(study_name)
    
    # Check if this trial count has been reviewed or dismissed
    already_reviewed = latest is not None and latest.get("trials_evaluated") == n_eval
    
    already_dismissed = False
    with get_db_session() as session:
        status_row = session.query(StudyStatus).filter_by(study_name=study_name).first()
        if status_row and status_row.nudge_dismissed_trials is not None:
            if status_row.nudge_dismissed_trials == n_eval:
                already_dismissed = True

    if already_reviewed or already_dismissed:
        review_recommended = False

    return {
        "review_recommended": review_recommended,
        "health_tier": health_tier,
        "health_reason": health_reason,
        "reasons": [{"code": health_tier, "message": health_reason}] if health_reason else [],
        "trials_evaluated": n_eval,
        "already_reviewed": already_reviewed,
        "already_dismissed": already_dismissed,
        "last_review_trials_evaluated": latest.get("trials_evaluated") if latest else None,
    }


# --- Review persistence (idempotent) ---
def get_latest_study_review(study_name: str) -> Optional[Dict[str, Any]]:
    with get_db_session() as session:
        row = (
            session.query(StudyReview)
            .filter_by(study_name=study_name)
            .order_by(StudyReview.created_at.desc(), StudyReview.id.desc())
            .first()
        )
        return row.to_dict() if row else None


def get_recent_study_reviews(study_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    with get_db_session() as session:
        rows = (
            session.query(StudyReview)
            .filter_by(study_name=study_name)
            .order_by(StudyReview.created_at.desc(), StudyReview.id.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


def save_study_review(
    study_name: str,
    summary: str,
    *,
    health_rating: Optional[int] = None,
    policy_action: str = "no_change",
    model_version: str = "unspecified",
    prompt_strategy: str = "coordinator_review",
    reasons: Optional[List[Dict[str, Any]]] = None,
    trials_evaluated: int = 0,
    estimated_dice_improvement: Optional[float] = None,
    cited_best_trial: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Persist a coordinator review. Idempotent per trial window unless force=True."""
    if policy_action not in POLICY_ACTIONS:
        policy_action = "no_change"
    if health_rating is not None:
        try:
            health_rating = max(1, min(5, int(health_rating)))
        except (TypeError, ValueError):
            health_rating = None
    if estimated_dice_improvement is not None:
        try:
            estimated_dice_improvement = float(estimated_dice_improvement)
        except (TypeError, ValueError):
            estimated_dice_improvement = None

    # Load study first to run validation assertions
    study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed_count = len(completed_trials)

    if completed_count < MIN_COMPLETED_FOR_FIRST_REVIEW:
        estimated_dice_improvement = -1.0

    baseline_best_score = get_best_primary_score(study)
    if baseline_best_score is None:
        baseline_best_score = -1.0

    if policy_action == "no_change":
        outcome_status = "not_applicable"
    else:
        outcome_status = "pending"

    # 1. Require at least one evaluated trial (ValueError, not assert, so it survives `python -O`).
    if trials_evaluated <= 0:
        raise ValueError("No trials have been evaluated yet. Cannot save review.")
    
    # 2. Verify trials_evaluated matches finished-trial idempotency window (COMPLETE+PRUNED+FAIL)
    evaluated_count = count_evaluated_trials(study)
    if trials_evaluated != evaluated_count:
        raise ValueError(
            f"Idempotency key trials_evaluated ({trials_evaluated}) must match "
            f"actual count of evaluated trials ({evaluated_count})."
        )

    # 3. Compare cited trial score with actual best trial score
    confidence = "high"
    if completed_trials:
        if len(study.directions) > 1:
            best_t = max(completed_trials, key=lambda t: t.values[1] if (t.values and len(t.values) > 1) else -float('inf'))
            actual_best_score = best_t.values[1] if (best_t.values and len(best_t.values) > 1) else 0.0
        else:
            best_t = study.best_trial
            actual_best_score = best_t.value

        if cited_best_trial is not None:
            cited_t = None
            for t in completed_trials:
                if t.number == cited_best_trial:
                    cited_t = t
                    break
            if cited_t:
                cited_score = (
                    cited_t.values[1]
                    if len(study.directions) > 1 and cited_t.values and len(cited_t.values) > 1
                    else cited_t.value
                )
                if cited_score is not None and actual_best_score is not None:
                    if abs(actual_best_score - cited_score) > 0.10:
                        confidence = "low"
            else:
                # Cited a non-existent completed trial
                confidence = "low"
        else:
            # Did not specify cited best trial
            confidence = "low"

    latest = get_latest_study_review(study_name)
    if latest and not force and latest.get("trials_evaluated") == trials_evaluated:
        return {"success": True, "duplicate": True, "review": latest}

    with get_db_session() as session:
        review = StudyReview(
            study_name=study_name,
            health_rating=health_rating,
            summary=summary,
            policy_action=policy_action,
            model_version=model_version or "unspecified",
            prompt_strategy=prompt_strategy or "coordinator_review",
            trials_evaluated=trials_evaluated,
            estimated_score_improvement=estimated_dice_improvement,
            cited_best_trial=cited_best_trial,
            confidence=confidence,
            baseline_best_score=baseline_best_score,
            outcome_status=outcome_status,
        )
        review.set_reasons(reasons)
        session.add(review)

        # Record review window without masking underlying study health
        status = session.query(StudyStatus).filter_by(study_name=study_name).first()
        if status:
            tier, reason = compute_health_tier(study, study_name)
            status.health_tier = tier
            status.health_reason = reason

        session.flush()
        saved = review.to_dict()
    return {"success": True, "duplicate": False, "review": saved}


# --- fANOVA + packet assembly ---
def _fanova_importances(study, config: Dict[str, Any]) -> Dict[str, float]:
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(complete) < 2:
        return {}
    try:
        if len(study.directions) > 1:
            importances = optuna.importance.get_param_importances(
                study,
                target=lambda t: t.values[1],
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )
        else:
            importances = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator()
            )
    except Exception:
        return {}

    aliases = config.get("legacy_param_aliases", {})
    display: Dict[str, float] = {}
    for param, value in importances.items():
        canonical = aliases.get(param, param)
        label = param_display_name(canonical, config)
        display[label] = max(display.get(label, 0.0), float(value))
    return display


def _recent_trials_summary(study, config: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    dice_fixed_key = ev.get("fixed_dice_attr", "dice_eval_fixed")
    bce_fixed_key = ev.get("fixed_bce_attr", "bce_eval_fixed")

    ordered = sorted(study.trials, key=lambda t: t.number, reverse=True)[:limit]
    out: List[Dict[str, Any]] = []
    for t in ordered:
        dice = t.values[1] if (t.state == TrialState.COMPLETE and t.values and len(t.values) > 1) else t.user_attrs.get("latest_dice")
        bce = t.values[0] if (t.state == TrialState.COMPLETE and t.values and len(t.values) > 1) else t.user_attrs.get("latest_bce")
        out.append({
            "number": t.number,
            "state": t.state.name,
            "params": normalize_trial_params(dict(t.params), config),
            "train_resolution": trial_train_resolution(t, train_param),
            "dice_train": dice,
            "bce_train": bce,
            "dice_eval_fixed": t.user_attrs.get(dice_fixed_key),
            "bce_eval_fixed": t.user_attrs.get(bce_fixed_key),
            "latest_epoch": t.user_attrs.get("latest_epoch"),
        })
    return out


def build_review_packet(study_name: str) -> Dict[str, Any]:
    """Assemble the compacted HPO review packet, utilizing a lazy materialization cache layer.
    
    This is the single source of truth for both the HTTP broker API and the MCP server.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        n_eval = count_evaluated_trials(study)

        # Check compacted packets cache
        with get_db_session() as session:
            cached = session.query(CompactedPacket).filter_by(
                study_name=study_name, trials_evaluated=n_eval
            ).first()
            if cached:
                try:
                    packet = json.loads(cached.packet_json)
                    n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
                    packet["statistical_confidence"] = compute_statistical_confidence(n_complete)
                    packet["coordinator_accuracy"] = compute_coordinator_accuracy(study_name)
                    packet["review_prompt"] = build_review_prompt(study_name)
                    packet["policy_actions"] = list(POLICY_ACTIONS)
                    packet["latest_review"] = get_latest_study_review(study_name)
                    return packet
                except Exception:
                    pass

        # Otherwise materialize from scratch
        # 1. Fetch DB metrics
        db_metrics = {}
        with get_db_session() as session:
            rows = session.query(TrialResult).filter_by(study_name=study_name).all()
            for r in rows:
                db_metrics[r.trial_id] = r.to_dict()

        # 2. Fetch search space and config
        search_space = load_active_search_space(study_name)
        config = load_hpo_config(study_name)

        # 3. Fetch project context
        project_context = {}
        with get_db_session() as session:
            context_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="project_context"
            ).first()
            if context_row:
                try:
                    project_context = json.loads(context_row.config_value)
                except Exception:
                    pass

        # 4. Fetch health tier
        health_tier, health_reason = compute_health_tier(study, study_name)

        # 5. Fetch past reviews
        past_reviews = []
        with get_db_session() as session:
            rows = (
                session.query(StudyReview)
                .filter_by(study_name=study_name)
                .order_by(StudyReview.created_at.desc(), StudyReview.id.desc())
                .limit(3)
                .all()
            )
            for r in rows:
                past_reviews.append(r.to_dict())

        # 6. Accuracy + statistical confidence
        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        statistical_confidence = compute_statistical_confidence(n_complete)
        accuracy_stats = compute_coordinator_accuracy(study_name)

        # 7. Assemble compacted packet using build_compacted_packet
        from .analytics import build_compacted_packet
        packet = build_compacted_packet(
            study_name,
            study,
            db_metrics,
            search_space,
            config,
            health_tier,
            health_reason,
            past_reviews,
            accuracy_stats,
            project_context,
            statistical_confidence,
        )

        # Cache it in compacted_packets table
        with get_db_session() as session:
            session.merge(CompactedPacket(
                study_name=study_name,
                trials_evaluated=n_eval,
                packet_json=json.dumps(packet)
            ))

        packet["statistical_confidence"] = statistical_confidence
        packet["coordinator_accuracy"] = accuracy_stats
        packet["review_prompt"] = build_review_prompt(study_name)
        packet["policy_actions"] = list(POLICY_ACTIONS)
        packet["latest_review"] = get_latest_study_review(study_name)
        return packet
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Failed to build review packet: {str(e)}"}

from pydantic import BaseModel, Field, field_validator

class UNetHyperparameters(BaseModel):
    learning_rate: float = Field(..., ge=1e-6, le=1e-1)
    batch_size: int = Field(..., ge=2, le=128)
    resolution: int = Field(..., ge=128, le=2048)
    model_capacity: str = Field(..., pattern="^(narrow|wide)$")
    loss_weight_ratio: float = Field(..., ge=0.0, le=1.0)

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError("Resolution must be a multiple of 32 for U-Net downsampling compatibility.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in [2, 4, 8, 16, 32, 64, 128]:
            raise ValueError("Batch size must be a power of 2 (e.g. 2, 4, 8, 16, 32, 64, 128).")
        return v

LEGACY_UNET_PARAMS = {
    "learning_rate",
    "batch_size",
    "resolution",
    "model_capacity",
    "loss_weight_ratio",
}

def _validate_manual_parameters(manual_parameters: Dict[str, Any], study_name: str) -> Dict[str, Any]:
    """Validate agent-proposed params against DB-backed search space."""
    config = load_hpo_config(study_name)
    norm = normalize_trial_params(dict(manual_parameters), config)

    space = load_active_search_space(study_name)
    space_keys = {k for k in space.keys() if not k.startswith("_") and isinstance(space.get(k), dict)}

    # Legacy fallback check
    if space_keys == LEGACY_UNET_PARAMS:
        try:
            valid = UNetHyperparameters(**norm)
            return {"ok": True, "params": valid.model_dump(), "error": None, "warnings": []}
        except Exception as exc:
            return {"ok": False, "params": {}, "error": str(exc), "warnings": []}

    warnings: List[str] = []
    out: Dict[str, Any] = {}

    for name, spec in space.items():
        if name.startswith("_") or not isinstance(spec, dict):
            continue
        ptype = spec.get("type", "float")
        if name not in norm:
            return {"ok": False, "params": {}, "error": f"Missing required parameter '{name}'.", "warnings": warnings}
        value = norm[name]

        if ptype in ("float", "float_log", "int"):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return {"ok": False, "params": {}, "error": f"Parameter '{name}' must be numeric.", "warnings": warnings}
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and num < float(lo):
                return {"ok": False, "params": {}, "error": f"'{name}'={num} below min {lo}.", "warnings": warnings}
            if hi is not None and num > float(hi):
                return {"ok": False, "params": {}, "error": f"'{name}'={num} above max {hi}.", "warnings": warnings}
            out[name] = int(round(num)) if ptype == "int" else num
        elif ptype == "categorical":
            options = spec.get("options", [])
            coerced = value
            if value not in options:
                for opt in options:
                    if str(opt) == str(value):
                        coerced = opt
                        break
            if coerced not in options:
                return {"ok": False, "params": {}, "error": f"'{name}'={value} not in options {options}.", "warnings": warnings}
            active = spec.get("active", options)
            if coerced not in active:
                warnings.append(f"'{name}'={coerced} is allowed but not in the active set {active}.")
            out[name] = coerced
        else:
            out[name] = value

    extra = [k for k in norm if k not in space_keys]
    if extra:
        warnings.append(f"Ignoring parameters not in search space: {extra}.")

    return {"ok": True, "params": out, "error": None, "warnings": warnings}


def load_study_cards(study_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Query and return generated study cards, loading their markdown contents from disk if available."""
    from .schema import StudyCard
    with get_db_session() as session:
        query = session.query(StudyCard)
        if study_name:
            query = query.filter_by(study_name=study_name)
        cards = query.all()
        
        result = []
        for c in cards:
            content = ""
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            full_path = os.path.join(root_dir, c.file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                except Exception:
                    pass
            card_dict = c.to_dict()
            card_dict["markdown_content"] = content
            result.append(card_dict)
        return result


_last_hook_trigger: Dict[str, float] = {}

def write_ide_status_file(study_name: str, health_tier: str, health_reason: str, study) -> None:
    """Writes the current health status to .hpo_status.json in the workspace root."""
    import datetime
    
    trials_evaluated = count_evaluated_trials(study)
    
    # Form status payload
    payload = {
        "study_name": study_name,
        "health_tier": health_tier.lower(),
        "health_reason": health_reason,
        "trials_evaluated": trials_evaluated,
        "review_recommended": health_tier.lower() in ("watch", "intervene"),
        "last_updated": datetime.datetime.utcnow().isoformat()
    }
    
    # Write to .hpo_status.json in workspace root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    status_file_path = os.path.join(root_dir, ".hpo_status.json")
    
    try:
        with open(status_file_path, "w") as f:
            json.dump(payload, f, indent=4)
    except Exception as e:
        print(f"Error writing .hpo_status.json: {e}")


