import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from typing import Dict, Any, Optional

from src.hpo_client import TrialSession

def simulate_training_epoch(
    epoch: int,
    params: Dict[str, Any]
) -> tuple[float, float]:
    """
    Simulates a training epoch on a segmentation model.
    Defines a continuous non-linear optimization landscape:
    - Optimal learning rate: log10(lr) = -3 (1e-3)
    - Optimal BCE weight ratio: 0.3
    - Resolution 1024 captures fine details better (higher Score), but takes longer
    - model_capacity 'wide' performs best
    """
    # This simulates a BCE/Dice optimization landscape with a known optimum at lr≈1e-3, resolution=1024
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
    cap = params.get("model_capacity", "narrow")
    if cap in ("wide", "resnet50"):
        enc_perf = 0.04
    elif cap in ("narrow", "resnet34", "efficientnet-b0"):
        enc_perf = 0.02
    else:
        enc_perf = 0.0

    # 4. Loss weight ratio performance (Peak at 0.3 BCE weight)
    loss_perf = -0.4 * (params["loss_weight_ratio"] - 0.3)**2

    # Assemble base Dice score ceiling (maximum is ~0.92)
    base_score = 0.70 + lr_perf + res_perf + enc_perf + loss_perf
    base_score = max(0.15, min(0.92, base_score))

    # Learning curve: approaches the ceiling asymptotically over epochs (max 10)
    progress = 1.0 - np.exp(-0.35 * epoch)
    current_score = base_score * progress

    # Add stochastic noise to simulate realistic batch training variances
    noise = np.random.normal(0, 0.008)
    current_score = float(max(0.01, min(0.95, current_score + noise)))

    # BCE Loss correlates inversely with Dice Score
    current_loss = float(max(0.02, 2.5 * (1.0 - current_score) + np.random.normal(0, 0.015)))

    return current_score, current_loss


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
        final_score = 0.0
        final_loss = 999.0
        final_score_fixed = 0.0
        final_loss_fixed = 999.0
        pruned = False
        
        # Prepare parameters for fixed resolution evaluation (fixed at 512px)
        params_fixed = dict(params)
        params_fixed["resolution"] = 512

        # 2. Run Epoch training loop
        for epoch in range(1, epochs_per_trial + 1):
            # Simulate training/val forward pass
            score, loss = simulate_training_epoch(epoch, params)
            score_fixed, loss_fixed = simulate_training_epoch(epoch, params_fixed)
            
            val_history.append({
                "epoch": epoch,
                "score": score,
                "loss": loss,
                "score_eval_fixed": score_fixed,
                "loss_eval_fixed": loss_fixed
            })
            
            print(f"  Epoch {epoch:02d}/{epochs_per_trial:02d} | Score (train): {score:.4f} | Loss: {loss:.4f} | Score (fixed eval @512px): {score_fixed:.4f}")
            
            # Record final metrics
            final_score = score
            final_loss = loss
            final_score_fixed = score_fixed
            final_loss_fixed = loss_fixed
            
            # 3. Intermediate epoch reporting & pruning evaluation
            try:
                should_prune = session.report_epoch(
                    epoch=epoch,
                    score=score,
                    loss=loss,
                    score_eval_fixed=score_fixed,
                    loss_eval_fixed=loss_fixed
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
                        score=score,
                        loss=loss,
                        score_eval_fixed=score_fixed,
                        loss_eval_fixed=loss_fixed,
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
                    score=final_score,
                    loss=final_loss,
                    score_eval_fixed=final_score_fixed,
                    loss_eval_fixed=final_loss_fixed,
                    weights_path=weights_path,
                    history=val_history,
                    state="COMPLETE"
                )
                print(f"  >> Trial {trial_id} Completed successfully!")
            except Exception as comp_err:
                print(f"Error completing trial: {comp_err}")

    print("\n=== Training Worker Execution Cycle Finished ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulated Pathfinder Training Worker")
    parser.add_argument("--study_name", default="demo_study", help="Study name")
    parser.add_argument("--max_trials", type=int, default=5, help="Number of trials to run")
    parser.add_argument("--epochs_per_trial", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--broker_url", default=None, help="Broker URL")
    args = parser.parse_args()

    run_training_worker(
        study_name=args.study_name,
        max_trials=args.max_trials,
        epochs_per_trial=args.epochs_per_trial,
        broker_url=args.broker_url
    )
