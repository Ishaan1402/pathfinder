import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from typing import Dict, Any, Optional

from src.hpo_client import TrialSession

def simulate_unet_training_epoch(
    epoch: int,
    params: Dict[str, Any]
) -> tuple[float, float]:
    """
    Simulates a U-Net training epoch on crack segmentation.
    Defines a continuous non-linear optimization landscape:
    - Optimal learning rate: log10(lr) = -3 (1e-3)
    - Optimal BCE weight ratio: 0.3
    - Resolution 1024 captures fine details better (higher Dice), but takes longer
    - model_capacity 'wide' performs best
    """
    # 1. LR performance (quadratic curve in log space)
    log_lr = np.log10(params["learning_rate"])
    lr_perf = -1.5 * (log_lr - (-3.0))**2  # Peak at -3.0 (1e-3)

    # 2. Resolution performance
    res_perf = 0.0
    if params["resolution"] == 1024:
        res_perf = 0.12
    elif params["resolution"] == 512:
        res_perf = 0.06

    # 3. Model capacity (wide vs narrow U-Net channel widths)
    cap = params.get("model_capacity") or params.get("encoder_name", "narrow")
    if cap in ("wide", "resnet50"):
        enc_perf = 0.04
    elif cap in ("narrow", "resnet34", "efficientnet-b0"):
        enc_perf = 0.02
    else:
        enc_perf = 0.0

    # 4. Loss weight ratio performance (Peak at 0.3 BCE weight)
    loss_perf = -0.4 * (params["loss_weight_ratio"] - 0.3)**2

    # Assemble base Dice score ceiling (maximum is ~0.92)
    base_dice = 0.70 + lr_perf + res_perf + enc_perf + loss_perf
    base_dice = max(0.15, min(0.92, base_dice))

    # Learning curve: approaches the ceiling asymptotically over epochs (max 10)
    progress = 1.0 - np.exp(-0.35 * epoch)
    current_dice = base_dice * progress

    # Add stochastic noise to simulate realistic batch training variances
    noise = np.random.normal(0, 0.008)
    current_dice = float(max(0.01, min(0.95, current_dice + noise)))

    # BCE Loss correlates inversely with Dice Score
    current_bce = float(max(0.02, 2.5 * (1.0 - current_dice) + np.random.normal(0, 0.015)))

    return current_dice, current_bce


def run_training_worker(
    study_name: str,
    agent_model: str = "optuna-tpe",
    prompt_strategy: str = "tpe_sampler",
    max_trials: int = 5,
    epochs_per_trial: int = 10,
    broker_url: Optional[str] = None
):
    """
    Decoupled Worker loop using TrialSession:
    1. Proposes/suggests next trial from the HTTP broker.
    2. Runs training simulation.
    3. Handles intermediate performance checks for pruning.
    4. Completes trial state and writes metrics.
    """
    print(f"=== Starting Training Worker for Study: '{study_name}' ===")
    broker_env = broker_url or os.getenv("HPO_BROKER_URL", "http://localhost:8000")
    print(f"Connecting to broker URL: {broker_env}\n")

    session = TrialSession(broker_url=broker_env, study_name=study_name)

    for i in range(max_trials):
        print(f"\n--- [Trial Suggestion Request {i+1}/{max_trials}] ---")
        
        # 1. Ask broker for next suggestion
        reasoning = f"Exploring hyperparameter space for U-Net encoder at resolution tradeoffs. Proposing trial {i+1}."
        
        try:
            suggestion = session.suggest(
                agent_model=agent_model,
                prompt_strategy=prompt_strategy,
                reasoning=reasoning,
                estimated_score_improvement=0.02
            )
        except Exception as exc:
            print(f"Failed to get suggestion from broker: {exc}")
            break
            
        trial_id = suggestion["trial_id"]
        params = suggestion["params"]
        
        print(f"Received Trial ID: {trial_id}")
        print(f"Hyperparameters: {params}")

        # OOM Safety Guardrail Simulation:
        # High resolution (1024) combined with a high batch size (>= 16) triggers a CUDA OOM.
        if params.get("resolution") == 1024 and params.get("batch_size", 0) >= 16:
            print("CUDA Out Of Memory simulated! (Resolution 1024 is incompatible with Batch Size >= 16).")
            # Mark trial as failed in Optuna via HTTP broker complete call
            try:
                session.complete(
                    epoch=0,
                    score=0.0,
                    loss=999.0,
                    state="FAIL",
                    oom_triggered=True
                )
                print(f"Trial {trial_id} marked as FAILED via broker due to OOM.")
            except Exception as tell_err:
                print(f"Error marking OOM failure: {tell_err}")
            continue

        # Initialize tracking metrics
        val_history = []
        final_dice = 0.0
        final_bce = 999.0
        pruned = False
        
        # 2. Run Epoch training loop
        for epoch in range(1, epochs_per_trial + 1):
            # Simulate training/val forward pass
            dice, bce = simulate_unet_training_epoch(epoch, params)
            val_history.append({"epoch": epoch, "dice": dice, "bce": bce})
            
            print(f"  Epoch {epoch:02d}/{epochs_per_trial:02d} | Dice: {dice:.4f} | BCE: {bce:.4f}")
            
            # Record final metrics
            final_dice = dice
            final_bce = bce
            
            # 3. Intermediate epoch reporting & pruning evaluation
            try:
                should_prune = session.report_epoch(
                    epoch=epoch,
                    score=dice,
                    loss=bce
                )
            except Exception as rep_err:
                print(f"Error reporting epoch: {rep_err}")
                should_prune = False

            if should_prune:
                print(f"  >> Trial {trial_id} is performing poorly. PRUNED at epoch {epoch}!")
                pruned = True

                try:
                    session.complete(
                        epoch=epoch,
                        score=dice,
                        loss=bce,
                        state="PRUNED"
                    )
                except Exception as prune_err:
                    print(f"Error marking pruned state: {prune_err}")
                break

            # Simulate training delay
            time.sleep(0.1)

        # 4. Save metrics & complete trial in registry if it ran to completion
        if not pruned:
            weights_path = f"checkpoints/trial_{trial_id}_res_{params['resolution']}_weights.pt"
            try:
                comp_result = session.complete(
                    epoch=epochs_per_trial,
                    score=final_dice,
                    loss=final_bce,
                    weights_path=weights_path,
                    history=val_history,
                    state="COMPLETE"
                )
                print(f"  >> Trial {trial_id} Completed successfully!")
            except Exception as comp_err:
                print(f"Error completing trial: {comp_err}")

    print("\n=== Training Worker Execution Cycle Finished ===")


if __name__ == "__main__":
    # Runs a default study local simulation of 5 trials
    run_training_worker(study_name="unet_crack_segmentation", max_trials=5)
