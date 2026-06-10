import math
import json
from typing import List, Dict, Any, Optional
from optuna.trial import TrialState
from .metrics import get_score, get_loss

def compress_loss_curve(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compresses a raw epoch-by-epoch history curve into key indicators to save tokens."""
    if not history:
        return {}
    scores = [h.get("score") for h in history if h.get("score") is not None]
    losses = [h.get("loss") for h in history if h.get("loss") is not None]
    
    res = {
        "initial_score": scores[0] if scores else None,
        "min_loss": min(losses) if losses else None,
        "final_score": scores[-1] if scores else None,
        "final_loss": losses[-1] if losses else None,
        "total_epochs": len(history)
    }
    
    # Linear slope of the last 10% of epochs (convergence slope)
    y = losses if losses else (scores if scores else [])
    if len(y) >= 2:
        n = max(2, int(len(y) * 0.1))
        last_y = y[-n:]
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(last_y)
        sum_xx = sum(xi * xi for xi in x)
        sum_xy = sum(xi * yi for xi, yi in zip(x, last_y))
        denom = n * sum_xx - sum_x * sum_x
        if denom != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        else:
            slope = 0.0
        res["convergence_slope"] = slope
    else:
        res["convergence_slope"] = 0.0
    return res

def bin_trials(study, db_metrics: Dict[int, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Segments trials into Elite (top 10%), Noise Floor (middle 80%), and Failure modes."""
    trials = list(study.trials)
    completed_trials = []
    failed_trials = []
    
    for t in trials:
        if t.state == TrialState.COMPLETE:
            # Use generic helpers that derive indices from study directions
            score = get_score(t, study) or 0.0
            loss = get_loss(t, study) or 0.0

            # Fetch from db_metrics if present
            metric = db_metrics.get(t.number, {})
            score = metric.get("primary_score") if metric.get("primary_score") is not None else score
            loss = metric.get("primary_loss") if metric.get("primary_loss") is not None else loss
            
            completed_trials.append({
                "trial_id": t.number,
                "params": dict(t.params),
                "primary_score": score,
                "primary_loss": loss,
                "epoch_reached": metric.get("epoch_reached", t.user_attrs.get("latest_epoch", 0))
            })
        elif t.state in (TrialState.FAIL, TrialState.PRUNED):
            # Check OOM or failure status
            metric = db_metrics.get(t.number, {})
            oom = metric.get("oom_triggered") or t.user_attrs.get("oom_triggered", False)
            failure_tag = metric.get("failure_tag") or ("OOM" if oom else "PRUNED" if t.state == TrialState.PRUNED else "FAILED")
            
            failed_trials.append({
                "trial_id": t.number,
                "params": dict(t.params),
                "failure_tag": failure_tag
            })
            
    # Sort completed trials by score descending (higher score is better)
    completed_trials.sort(key=lambda x: x["primary_score"] or 0.0, reverse=True)
    n_completed = len(completed_trials)
    
    elite = []
    noise_floor = {}
    
    if n_completed > 0:
        elite_count = max(1, int(math.ceil(n_completed * 0.1)))
        elite = completed_trials[:elite_count]
        middle_trials = completed_trials[elite_count:]
        
        if middle_trials:
            scores = [t["primary_score"] for t in middle_trials if t["primary_score"] is not None]
            n_mid = len(scores)
            if n_mid > 0:
                mean_score = sum(scores) / n_mid
                variance = sum((x - mean_score) ** 2 for x in scores) / n_mid
                sorted_scores = sorted(scores)
                median_score = sorted_scores[n_mid // 2]
            else:
                median_score = 0.0
                variance = 0.0
                
            param_ranges = {}
            # Gather all param names in search space
            param_names = set(search_space.keys())
            for t in middle_trials:
                param_names.update(t["params"].keys())
                
            for p_name in param_names:
                p_vals = [t["params"].get(p_name) for t in middle_trials if t["params"].get(p_name) is not None]
                if not p_vals:
                    continue
                try:
                    # Numeric ranges
                    numeric_vals = [float(v) for v in p_vals]
                    param_ranges[p_name] = [min(numeric_vals), max(numeric_vals)]
                except (ValueError, TypeError):
                    # Categorical counts
                    counts = {}
                    for v in p_vals:
                        counts[str(v)] = counts.get(str(v), 0) + 1
                    param_ranges[p_name] = counts
                    
            noise_floor = {
                "count": len(middle_trials),
                "median_score": median_score,
                "score_variance": variance,
                "param_ranges": param_ranges
            }
        else:
            noise_floor = {
                "count": 0,
                "median_score": 0.0,
                "score_variance": 0.0,
                "param_ranges": {}
            }
            
    # Aggregated failure/OOM modes count matrix
    failure_matrix = {}
    for t in failed_trials:
        # Construct combinations of parameters for categorization
        # Focus on key hyperparams: batch_size, resolution, lr if they exist, or just all params sorted
        important_params = ["batch_size", "resolution", "lr"]
        key_parts = []
        for p in important_params:
            if p in t["params"]:
                key_parts.append(f"{p}={t['params'][p]}")
        
        if not key_parts:
            # Fallback to sorting all params
            key_parts = [f"{k}={v}" for k, v in sorted(t["params"].items())]
            
        param_key = " && ".join(key_parts) if key_parts else "unknown_params"
        tag = t["failure_tag"]
        
        full_key = f"{param_key} [{tag}]"
        failure_matrix[full_key] = failure_matrix.get(full_key, 0) + 1
        
    return {
        "elite": elite,
        "noise_floor": noise_floor,
        "failure_matrix": failure_matrix
    }

def build_compacted_packet(
    study_name: str,
    study,
    db_metrics: Dict[int, Any],
    search_space: Dict[str, Any],
    config: Dict[str, Any],
    health_tier: str,
    health_reason: Optional[str],
    past_reviews: List[Dict[str, Any]],
    accuracy_stats: Dict[str, Any],
    project_context: Dict[str, Any],
    statistical_confidence: str = "low",
) -> Dict[str, Any]:
    """Assembles a highly compressed, token-efficient HPO review packet."""
    from .hpo_coordinator import (
        _fanova_importances,
        pareto_trial_numbers_deploy_aware,
        check_boundary_hits,
        compute_fidelity_durations,
        compute_vram_telemetry,
        compute_spearman_rank_correlation
    )
    
    trials = list(study.trials)
    
    # Study status counts
    counts = {
        "total": len(trials),
        "complete": sum(1 for t in trials if t.state == TrialState.COMPLETE),
        "pruned": sum(1 for t in trials if t.state == TrialState.PRUNED),
        "failed": sum(1 for t in trials if t.state == TrialState.FAIL),
        "running": sum(1 for t in trials if t.state == TrialState.RUNNING),
    }
    
    # 1. Compacted Trial Bins
    trial_bins = bin_trials(study, db_metrics, search_space)
    
    # 2. fANOVA Importances (Top 5 only)
    raw_importances = _fanova_importances(study, config)
    sorted_importances = sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    top_fanova = dict(sorted_importances)
    
    # 3. Spearman correlations (with confidence tags, numeric params only)
    spearman_correlations = {}
    complete_trials = [t for t in trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        dice_fixed_key = config.get("eval_protocol", {}).get("fixed_dice_attr", "dice_eval_fixed")
        for p_name, p_info in search_space.items():
            # Only analyze numeric parameters
            if p_info.get("type") not in ("float", "float_log", "int"):
                continue
                
            paired_x = []
            paired_y = []
            try:
                for t in complete_trials:
                    val = t.params.get(p_name)
                    if val is None:
                        continue
                    
                    # Score lookup
                    fd = t.user_attrs.get(dice_fixed_key)
                    if fd is not None:
                        score_val = float(fd)
                    else:
                        s = get_score(t, study)
                        score_val = float(s) if s is not None else 0.0
                        
                    paired_x.append(float(val))
                    paired_y.append(score_val)
            except (ValueError, TypeError):
                continue
                
            if len(set(paired_x)) > 1 and len(paired_x) >= 3:
                n_samples = len(paired_x)
                confidence = "Low" if n_samples < 8 else "Moderate" if n_samples < 15 else "High"
                corr_coef = compute_spearman_rank_correlation(paired_x, paired_y)
                spearman_correlations[p_name] = {
                    "coefficient": round(corr_coef, 4),
                    "n_samples": n_samples,
                    "confidence": confidence
                }

    # 4. Boundary hits (Pareto-adjacent params only)
    pareto_numbers = (
        pareto_trial_numbers_deploy_aware(study, config) if len(study.directions) > 1 else []
    )
    raw_boundary_hits = check_boundary_hits(study, pareto_numbers, search_space)
    # Filter to only params that actually hit boundaries (near_min or near_max)
    boundary_hits = {k: v for k, v in raw_boundary_hits.items() if v.get("hit_ratio", 0) > 0.0}

    # 5. Fidelity durations
    fidelity_durations = compute_fidelity_durations(study, config)

    # 6. VRAM Telemetry (with RSE prediction intervals)
    vram_telemetry = compute_vram_telemetry(study, db_metrics, search_space, config)

    # 7. Past 3 Reviews (Summary + Action + Rating only, not full text if long)
    compact_reviews = []
    for r in past_reviews[:3]:
        summary_lines = r.get("summary", "").split("\n")
        short_summary = summary_lines[0] if summary_lines else ""
        if len(r.get("summary", "")) > 150:
            short_summary = r.get("summary", "")[:147] + "..."
            
        compact_reviews.append({
            "id": r.get("id"),
            "created_at": r.get("created_at"),
            "health_rating": r.get("health_rating"),
            "policy_action": r.get("policy_action"),
            "trials_evaluated": r.get("trials_evaluated"),
            "estimated_score_improvement": r.get("estimated_score_improvement"),
            "quality_flagged": r.get("quality_flagged", False),
            "outcome_status": r.get("outcome_status"),
            "summary": short_summary
        })

    return {
        "study_name": study_name,
        "counts": counts,
        "project_context": project_context,
        "health": {
            "tier": health_tier,
            "reason": health_reason
        },
        "trial_bins": trial_bins,
        "fanova_importances": top_fanova,
        "spearman_correlations": spearman_correlations,
        "boundary_hits": boundary_hits,
        "fidelity_durations": fidelity_durations,
        "vram_telemetry": vram_telemetry,
        "past_reviews": compact_reviews,
        "coordinator_accuracy": accuracy_stats,
        "statistical_confidence": statistical_confidence,
        "metric_score_label": config.get("metric_score_label", "Score"),
        "metric_loss_label": config.get("metric_loss_label", "Loss")
    }
