import json
import math
import os
from typing import Any, Dict, List, Optional

import optuna
from optuna.trial import TrialState

from .db_manager import get_db_session, DATABASE_URL
from .hpo_config import load_hpo_config, param_display_name
from .metrics import get_score, get_loss, get_best_trial, score_objective_index, get_completed_trials, get_eval_attr_names
from .schema import TrialResult, SystemConfiguration, CompactedPacket, StudyCard

logger = __import__('logging').getLogger(__name__)


def compress_loss_curve(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    scores = [h.get("score") for h in history if h.get("score") is not None]
    losses = [h.get("loss") for h in history if h.get("loss") is not None]

    res = {
        "initial_score": scores[0] if scores else None,
        "min_loss": min(losses) if losses else None,
        "final_score": scores[-1] if scores else None,
        "final_loss": losses[-1] if losses else None,
        "total_epochs": len(history),
    }

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
    from .search_space import _fixed_categorical_params
    fixed = _fixed_categorical_params(search_space)
    trials = list(study.trials)
    completed_trials: List[Dict[str, Any]] = []
    failed_trials: List[Dict[str, Any]] = []

    for t in trials:
        if t.state == TrialState.COMPLETE:
            s = get_score(t, study)
            score = s if s is not None else 0.0
            l = get_loss(t, study)
            loss = l if l is not None else 0.0

            metric = db_metrics.get(t._trial_id, {})
            score = metric.get("primary_score") if metric.get("primary_score") is not None else score
            loss = metric.get("primary_loss") if metric.get("primary_loss") is not None else loss

            params = {**dict(t.params), **fixed}
            completed_trials.append({
                "trial_id": t.number,
                "params": params,
                "primary_score": score,
                "primary_loss": loss,
                "epoch_reached": metric.get("epoch_reached", t.user_attrs.get("latest_epoch", 0)),
            })
        elif t.state in (TrialState.FAIL, TrialState.PRUNED):
            metric = db_metrics.get(t._trial_id, {})
            oom = metric.get("oom_triggered") or t.user_attrs.get("oom_triggered", False)
            failure_tag = metric.get("failure_tag") or ("OOM" if oom else "PRUNED" if t.state == TrialState.PRUNED else "FAILED")
            failed_trials.append({
                "trial_id": t.number,
                "params": {**dict(t.params), **fixed},
                "failure_tag": failure_tag,
            })

    completed_trials.sort(key=lambda x: x["primary_score"] or 0.0, reverse=True)
    n_completed = len(completed_trials)

    elite: List[Dict[str, Any]] = []
    noise_floor: Dict[str, Any] = {}

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
            param_names = set(search_space.keys())
            for t in middle_trials:
                param_names.update(t["params"].keys())
            for p_name in param_names:
                p_vals = [t["params"].get(p_name) for t in middle_trials if t["params"].get(p_name) is not None]
                if not p_vals:
                    continue
                try:
                    numeric_vals = [float(v) for v in p_vals]
                    param_ranges[p_name] = [min(numeric_vals), max(numeric_vals)]
                except (ValueError, TypeError):
                    counts: Dict[str, int] = {}
                    for v in p_vals:
                        counts[str(v)] = counts.get(str(v), 0) + 1
                    param_ranges[p_name] = counts

            noise_floor = {
                "count": len(middle_trials),
                "median_score": median_score,
                "score_variance": variance,
                "param_ranges": param_ranges,
            }
        else:
            noise_floor = {"count": 0, "median_score": 0.0, "score_variance": 0.0, "param_ranges": {}}

    failure_matrix: Dict[str, int] = {}
    for t in failed_trials:
        important_params = ["batch_size", "resolution", "lr"]
        key_parts = []
        for p in important_params:
            if p in t["params"]:
                key_parts.append(f"{p}={t['params'][p]}")
        if not key_parts:
            key_parts = [f"{k}={v}" for k, v in sorted(t["params"].items())]
        param_key = " && ".join(key_parts) if key_parts else "unknown_params"
        tag = t["failure_tag"]
        full_key = f"{param_key} [{tag}]"
        failure_matrix[full_key] = failure_matrix.get(full_key, 0) + 1

    return {"elite": elite, "noise_floor": noise_floor, "failure_matrix": failure_matrix}


# --- Train-resolution helper (shared with pruning/pareto) ---

def trial_train_resolution(trial, train_param: str) -> Optional[int]:
    val = trial.params.get(train_param)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# --- fANOVA ---

def get_fanova_importances(study, config: Dict[str, Any]) -> Dict[str, float]:
    complete = get_completed_trials(study)
    if len(complete) < 2:
        return {}
    importances: Dict[str, float] = {}
    try:
        if len(study.directions) > 1:
            _si = score_objective_index(study)
            if _si is not None:
                importances = optuna.importance.get_param_importances(
                    study,
                    target=lambda t, idx=_si: t.values[idx] if (t.values and len(t.values) > idx) else None,
                    evaluator=optuna.importance.FanovaImportanceEvaluator(),
                )
        else:
            importances = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator()
            )
    except Exception:
        return {}

    display: Dict[str, float] = {}
    for param, value in importances.items():
        label = param_display_name(param, config)
        display[label] = max(display.get(label, 0.0), float(value))
    return display


# --- Pareto ---

def pareto_trial_numbers_deploy_aware(study, hpo_config: Dict[str, Any]) -> List[int]:
    ev = hpo_config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    low_warn = ev.get("low_train_res_warning")
    low_warn = int(low_warn) if low_warn is not None else None
    deploy_only = ev.get("pareto_deploy_resolution_only", True)
    score_fixed_key = ev.get("fixed_score_attr", "score_eval_fixed")

    points: List[tuple] = []
    for t in study.trials:
        if t.state != TrialState.COMPLETE:
            continue
        loss_val = get_loss(t, study)
        score_val = get_score(t, study)
        if loss_val is None or score_val is None:
            continue
        train_res = trial_train_resolution(t, train_param)
        if deploy_only and low_warn is not None and train_res is not None and train_res < low_warn:
            continue
        if ev.get("enabled"):
            fd = t.user_attrs.get(score_fixed_key)
            if fd is not None:
                score_val = float(fd)
        points.append((t.number, float(loss_val), float(score_val)))

    if not points:
        try:
            return [t.number for t in study.best_trials]
        except Exception:
            return []

    pareto: List[int] = []
    for num_i, loss_i, score_i in points:
        dominated = False
        for num_j, loss_j, score_j in points:
            if num_i == num_j:
                continue
            if loss_j <= loss_i and score_j >= score_i and (loss_j < loss_i or score_j > score_i):
                dominated = True
                break
        if not dominated:
            pareto.append(num_i)
    return pareto


# --- Boundary hits ---

def check_boundary_hits(study, pareto_numbers: List[int], search_space: Dict[str, Any]) -> Dict[str, Any]:
    hits: Dict[str, Any] = {}
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
                "bound_hit": "min" if near_min_count > near_max_count else "max" if near_max_count > near_min_count else "both",
            }
    return hits


# --- Fidelity durations ---

def compute_fidelity_durations(study, config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    groups: Dict[int, List] = {}
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

    res_stats: Dict[int, Dict[str, Any]] = {}
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
                "count": len(durations),
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
    return {"fidelity_param": train_param, "lowest_scale": lowest_scale, "scales": res_stats}


# --- VRAM telemetry ---

def fit_vram_model(trials: List, db_metrics: Dict[int, Any], train_param: str) -> Optional[Dict[str, Any]]:
    points: List[tuple] = []
    for t in trials:
        if t.state != TrialState.COMPLETE:
            continue
        metric = db_metrics.get(t._trial_id, {})
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

    if len(points) < 6:
        return None

    X = [p[0] * (p[1] ** 2) for p in points]
    Y = [p[2] for p in points]
    if len(set(X)) < 2:
        return None

    n = len(points)
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_xx = sum(x * x for x in X)
    sum_xy = sum(X[i] * Y[i] for i in range(n))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    ssr = sum((Y[i] - (slope * X[i] + intercept)) ** 2 for i in range(n))
    rse = (ssr / (n - 2)) ** 0.5 if n > 2 else 0.0
    return {"slope": slope, "intercept": intercept, "n_points": n, "rse": rse}


def compute_vram_telemetry(study, db_metrics: Dict[int, Any], search_space: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    trials = list(study.trials)
    model = fit_vram_model(trials, db_metrics, train_param)

    gpu_capacity_gb = 0.0
    gpu_models: List[str] = []
    oom_count = 0

    for t in trials:
        metric = db_metrics.get(t._trial_id, {})
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
                    "risk_level": "high" if predicted_max_vram > gpu_capacity_gb else "medium",
                }

    return {
        "gpu_model": gpu_model,
        "gpu_capacity_gb": gpu_capacity_gb,
        "oom_count": oom_count,
        "vram_model": model,
        "bounds_oom_risk": oom_risk,
    }


# --- Eval insights ---

def study_eval_insights(study, config: Dict[str, Any]) -> Dict[str, Any]:
    ev = config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    fixed_res = ev.get("fixed_resolution")
    low_warn = ev.get("low_train_res_warning")
    score_fixed_key, _ = get_eval_attr_names(ev)

    complete = get_completed_trials(study)
    by_res: Dict[int, List] = {}
    warnings: List[Dict[str, Any]] = []

    for t in complete:
        tr = t.params.get(train_param)
        if tr is not None:
            by_res.setdefault(int(tr), []).append(t)

    res_summary: Dict[int, Dict[str, Any]] = {}
    for res, trials in sorted(by_res.items()):
        scores = [get_score(t, study) for t in trials]
        scores = [s for s in scores if s is not None]
        fixed_scores = [t.user_attrs.get(score_fixed_key) for t in trials if t.user_attrs.get(score_fixed_key) is not None]
        res_summary[res] = {
            "count": len(trials),
            "best_score_train": max(scores) if scores else None,
            "best_score_fixed": max(fixed_scores) if fixed_scores else None,
        }

    valid_deploy_exists = False
    if low_warn is not None:
        for t in complete:
            tr = t.params.get(train_param)
            fd = t.user_attrs.get(score_fixed_key)
            if tr is not None and int(tr) >= int(low_warn) and fd is not None:
                valid_deploy_exists = True
                break

    if complete and ev.get("enabled") and fixed_res:
        best_train = get_best_trial(complete, study)
        if best_train is None:
            best_train = complete[0]
        train_res = best_train.params.get(train_param)
        if train_res is not None and low_warn and int(train_res) < int(low_warn) and not valid_deploy_exists:
            warnings.append({
                "code": "low_train_res_pareto",
                "trial_number": best_train.number,
                "message": f"Pareto-best trial #{best_train.number} trained at scale {train_res}, below warning threshold {low_warn}. Check fixed eval.",
            })
        fd = best_train.user_attrs.get(score_fixed_key)
        td = get_score(best_train, study)
        if fd is not None and td is not None and (td - fd) > 0.08:
            warnings.append({
                "code": "train_eval_gap",
                "trial_number": best_train.number,
                "message": f"Trial #{best_train.number}: train score {td:.3f} vs fixed-eval score {fd:.3f} — train scale/resolution may be inflating scores.",
            })

    best_deploy: Any = None
    if ev.get("enabled"):
        ranked = [t for t in complete if t.user_attrs.get(score_fixed_key) is not None]
        if ranked:
            best_deploy = max(ranked, key=lambda t: t.user_attrs.get(score_fixed_key))

    return {
        "resolution_summary": res_summary,
        "warnings": warnings,
        "best_deploy_trial_number": best_deploy.number if best_deploy else None,
        "best_deploy_score_fixed": best_deploy.user_attrs.get(score_fixed_key) if best_deploy else None,
    }


# --- Prune rate clusters ---

def compute_prune_rate_clusters(study, search_space: Dict[str, Any]) -> Dict[str, Any]:
    clusters: Dict[str, Any] = {}
    continuous_params = []
    for p_name, p_info in search_space.items():
        if p_info.get("type", "") in ("float", "float_log", "int"):
            continuous_params.append(p_name)

    for p_name in continuous_params:
        vals: List[float] = []
        states: List[TrialState] = []
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
            {"min": v_min + w, "max": v_min + 2 * w, "total": 0, "pruned": 0},
            {"min": v_min + 2 * w, "max": v_max, "total": 0, "pruned": 0},
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


# --- Study packet assembly ---

def build_compacted_packet(
    study_name: str,
    study,
    db_metrics: Dict[int, Any],
    search_space: Dict[str, Any],
    config: Dict[str, Any],
    health_tier: str,
    health_reason: Optional[str],
    project_context: Dict[str, Any],
    statistical_confidence: str = "low",
) -> Dict[str, Any]:
    trials = list(study.trials)

    counts = {
        "total": len(trials),
        "complete": sum(1 for t in trials if t.state == TrialState.COMPLETE),
        "pruned": sum(1 for t in trials if t.state == TrialState.PRUNED),
        "failed": sum(1 for t in trials if t.state == TrialState.FAIL),
        "running": sum(1 for t in trials if t.state == TrialState.RUNNING),
    }

    trial_bins = bin_trials(study, db_metrics, search_space)

    try:
        raw_importances = get_fanova_importances(study, config)
    except Exception:
        raw_importances = {}
    sorted_importances = sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    top_fanova = dict(sorted_importances)

    pareto_numbers = pareto_trial_numbers_deploy_aware(study, config) if len(study.directions) > 1 else []
    boundary_hits = check_boundary_hits(study, pareto_numbers, search_space)
    fidelity_durations = compute_fidelity_durations(study, config)
    vram_telemetry = compute_vram_telemetry(study, db_metrics, search_space, config)

    return {
        "study_name": study_name,
        "counts": counts,
        "project_context": project_context,
        "health": {"tier": health_tier, "reason": health_reason},
        "trial_bins": trial_bins,
        "fanova_importances": top_fanova,
        "boundary_hits": boundary_hits,
        "fidelity_durations": fidelity_durations,
        "vram_telemetry": vram_telemetry,
        "statistical_confidence": statistical_confidence,
        "metric_score_label": config.get("metric_score_label", "Score"),
        "metric_loss_label": config.get("metric_loss_label", "Loss"),
    }


def build_study_packet(study_name: str) -> Dict[str, Any]:
    try:
        from .health import compute_health_tier, compute_statistical_confidence, count_evaluated_trials

        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        n_eval = count_evaluated_trials(study)

        with get_db_session() as session:
            cached = session.query(CompactedPacket).filter_by(
                study_name=study_name, trials_evaluated=n_eval
            ).first()
            if cached:
                try:
                    packet = json.loads(cached.packet_json)
                    n_complete = len(get_completed_trials(study))
                    packet["statistical_confidence"] = compute_statistical_confidence(n_complete)
                    return packet
                except Exception as e:
                    logger.warning(f"Failed to load cached packet for {study_name}: {e}")

        with get_db_session() as session:
            db_metrics: Dict[int, Any] = {}
            rows = session.query(TrialResult).filter_by(study_name=study_name).all()
            for r in rows:
                db_metrics[r.trial_id] = r.to_dict()

            from .search_space import load_search_space
            search_space = load_search_space(study_name)
            config = load_hpo_config(study_name)

            project_context: Dict[str, Any] = {}
            context_row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="project_context"
            ).first()
            if context_row:
                try:
                    project_context = json.loads(context_row.config_value)
                except Exception as e:
                    logger.warning(f"Failed to parse project_context for {study_name}: {e}")

            health_tier, health_reason = compute_health_tier(study, study_name)

        n_complete = len(get_completed_trials(study))
        statistical_confidence = compute_statistical_confidence(n_complete)

        packet = build_compacted_packet(
            study_name, study, db_metrics, search_space, config,
            health_tier, health_reason, project_context, statistical_confidence,
        )

        with get_db_session() as session:
            session.merge(CompactedPacket(
                study_name=study_name,
                trials_evaluated=n_eval,
                packet_json=json.dumps(packet),
            ))

        packet["statistical_confidence"] = statistical_confidence
        return packet
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Failed to build study packet: {str(e)}"}


# --- Study cards ---

def load_study_cards(study_name: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db_session() as session:
        query = session.query(StudyCard)
        if study_name:
            query = query.filter_by(study_name=study_name)
        cards = query.all()

        result: List[Dict[str, Any]] = []
        for c in cards:
            card_dict = c.to_dict()
            full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), c.file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        card_dict["markdown_content"] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read study card {full_path}: {e}")
            else:
                card_dict["markdown_content"] = ""
            result.append(card_dict)
        return result
