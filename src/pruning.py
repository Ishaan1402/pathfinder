import numpy as np
from typing import Optional, Dict, Any, List
from optuna.trial import TrialState

from src.analytics import trial_train_resolution as _trial_train_resolution


def _effective_train_resolution(
    trial, hpo_config: Dict[str, Any], space: Dict[str, Any]
) -> Optional[Any]:
    train_param = hpo_config.get("eval_protocol", {}).get("train_resolution_param", "resolution")
    val = trial.params.get(train_param)
    if val is not None:
        return val
    val = trial.user_attrs.get(train_param)
    if val is not None:
        return val
    res_cfg = space.get(train_param) or space.get("resolution")
    if isinstance(res_cfg, dict) and res_cfg.get("type") == "categorical":
        active = res_cfg.get("active") or []
        if len(active) == 1:
            return active[0]
    return None


def _epoch_composite_score(study, trial, epoch: int, ev: Dict[str, Any]) -> Optional[float]:
    """Composite score at epoch, Z-score normalized against study rolling history at the same epoch, fallback to (score - loss)."""
    use_fixed = ev.get("enabled") and ev.get("use_fixed_metric_for_pruning")
    
    # 1. Retrieve current trial's metrics at this epoch
    curr_score = None
    curr_loss = None
    history = trial.user_attrs.get("history", [])
    if isinstance(history, list):
        for entry in history:
            if entry.get("epoch") == epoch:
                if use_fixed and entry.get("score_eval_fixed") is not None:
                    curr_score = entry.get("score_eval_fixed")
                    curr_loss = entry.get("loss_eval_fixed", entry.get("loss", 0.0))
                else:
                    curr_score = entry.get("score")
                    curr_loss = entry.get("loss")
                break
                
    if curr_score is None or curr_loss is None:
        if epoch in trial.intermediate_values:
            return float(trial.intermediate_values[epoch])
        return None

    # 2. Gather metrics at the SAME epoch across all Complete/Running trials
    scores = []
    losses = []
    for t in study.trials:
        if t.state not in (TrialState.COMPLETE, TrialState.RUNNING):
            continue
        t_history = t.user_attrs.get("history", [])
        if isinstance(t_history, list):
            for entry in t_history:
                if entry.get("epoch") == epoch:
                    if use_fixed and entry.get("score_eval_fixed") is not None:
                        s = entry.get("score_eval_fixed")
                        loss_val = entry.get("loss_eval_fixed", entry.get("loss", 0.0))
                    else:
                        s = entry.get("score")
                        loss_val = entry.get("loss")
                    if s is not None and loss_val is not None:
                        scores.append(float(s))
                        losses.append(float(loss_val))
                    break

    # 3. Z-score normalize if we have enough history (>= 10 values)
    if len(scores) < 10:
        # Not enough history: simple linear composite (score - loss)
        return float(curr_score) - float(curr_loss) if curr_loss is not None else float(curr_score)

    score_mean, score_std = np.mean(scores), np.std(scores)
    loss_mean, loss_std = np.mean(losses), np.std(losses)

    # Tweak 2: Clamp standard deviation to prevent division by zero in early homogeneous trials
    score_std = score_std if score_std > 1e-6 else 1.0
    loss_std = loss_std if loss_std > 1e-6 else 1.0

    z_score = (float(curr_score) - score_mean) / score_std
    z_loss = -(float(curr_loss) - loss_mean) / loss_std

    return float(z_score + z_loss)


def _pruning_peer_trials(study, trial_obj, hpo_config: Dict[str, Any]) -> List:
    ev = hpo_config.get("eval_protocol", {})
    train_param = ev.get("train_resolution_param", "resolution")
    current_res = _trial_train_resolution(trial_obj, train_param)
    low_warn = ev.get("low_train_res_warning")
    low_warn = int(low_warn) if low_warn is not None else None
    same_only = ev.get("prune_compare_same_resolution_only", True)
    exclude_low = ev.get("prune_exclude_low_res_from_baseline", True)

    peers = []
    for t in study.trials:
        if t.number == trial_obj.number:
            continue
        if t.state not in (TrialState.COMPLETE, TrialState.RUNNING):
            continue
        peer_res = _trial_train_resolution(t, train_param)
        if same_only and current_res is not None and peer_res is not None:
            if peer_res != current_res:
                continue
        if (
            exclude_low
            and low_warn is not None
            and current_res is not None
            and current_res >= low_warn
            and peer_res is not None
            and peer_res < low_warn
        ):
            continue
        peers.append(t)
    return peers
