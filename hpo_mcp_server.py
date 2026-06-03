import os
import json
import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import optuna
from optuna.trial import TrialState
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from mcp.server.fastmcp import FastMCP

from db_manager import init_db, get_db_session, DATABASE_URL
from schema import DatasetVersion, SegmentationMetric, AgentReasoningLog, InvalidProposal

# Initialize custom tables
init_db()

# Create FastMCP server
mcp = FastMCP("HPO Tuning Assistant")

# Pydantic Hyperparameter Validation Schema (Hallucination Guardrail)
class UNetHyperparameters(BaseModel):
    learning_rate: float = Field(..., ge=1e-6, le=1e-1)
    batch_size: int = Field(..., ge=2, le=128)
    resolution: int = Field(..., ge=128, le=2048)
    encoder_name: str = Field(..., pattern="^(resnet(18|34|50)|efficientnet-b[0-4]|unet_basic)$")
    loss_weight_ratio: float = Field(..., ge=0.0, le=1.0)  # Weight for BCE vs Dice

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


@mcp.tool()
def register_dataset_version(
    version_id: str,
    crack_surface_type: str,
    resolution_width: int,
    resolution_height: int,
    total_images: int
) -> str:
    """
    Registers a new dataset version slice in the database for tracking.
    """
    with get_db_session() as session:
        existing = session.query(DatasetVersion).filter_by(version_id=version_id).first()
        if existing:
            return f"Dataset version {version_id} already exists."
        
        dv = DatasetVersion(
            version_id=version_id,
            crack_surface_type=crack_surface_type,
            resolution_width=resolution_width,
            resolution_height=resolution_height,
            total_images=total_images
        )
        session.add(dv)
    return f"Dataset version {version_id} successfully registered."


@mcp.tool()
def create_new_study(study_name: str, multi_objective: bool = True) -> str:
    """
    Initializes a new Optuna study in the database.
    If multi_objective is True, optimizes both BCE (minimize) and Dice Score (maximize).
    If False, optimizes Dice Score (maximize) directly.
    """
    try:
        if multi_objective:
            study = optuna.create_study(
                study_name=study_name,
                storage=DATABASE_URL,
                directions=["minimize", "maximize"],
                load_if_exists=True
            )
            return f"Multi-objective study '{study_name}' initialized (Directions: minimize BCE, maximize Dice)."
        else:
            # We also configure a MedianPruner for single-objective to show ASHA / SHA early stopping
            study = optuna.create_study(
                study_name=study_name,
                storage=DATABASE_URL,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
                load_if_exists=True
            )
            return f"Single-objective study '{study_name}' initialized (Direction: maximize Dice. Pruner: MedianPruner)."
    except Exception as e:
        return f"Error creating study: {str(e)}"


@mcp.tool()
def suggest_next_trial(
    study_name: str,
    agent_model: str,
    prompt_strategy: str,
    reasoning: str,
    estimated_dice_improvement: float,
    manual_parameters: Optional[Dict[str, Any]] = None,
    dataset_version_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Suggests the next configuration to run. If manual_parameters are provided by the AI,
    validates them against Pydantic constraints (hallucination guardrail).
    Logs the AI's reasoning prior to the run.
    """
    try:
        # 1. Handle study loading
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        # 2. Check and validate manual parameters (if provided)
        if manual_parameters:
            try:
                valid_params = UNetHyperparameters(**manual_parameters)
                study.enqueue_trial(valid_params.model_dump())
            except Exception as val_err:
                # Log invalid proposal for AI profiling
                with get_db_session() as session:
                    ip = InvalidProposal(
                        study_name=study_name,
                        model_version=agent_model,
                        prompt_strategy=prompt_strategy,
                        validation_error=str(val_err)
                    )
                    ip.set_parameters(manual_parameters)
                    session.add(ip)
                return {
                    "success": False,
                    "error": "Parameter Validation Failed (Hallucination Guardrail)",
                    "details": str(val_err)
                }

        # 3. Ask Optuna for next trial parameters (TPE/conditional logic)
        # If a manual trial was enqueued, study.ask() will return it first.
        # Otherwise, TPE generates a new point.
        trial = study.ask()
        
        # Define TPE fallback distributions if not manually enqueued
        if not manual_parameters:
            # Conditional spaces logic: high resolution -> smaller batch sizes
            res = trial.suggest_categorical("resolution", [256, 512, 1024])
            if res == 1024:
                trial.suggest_categorical("batch_size", [2, 4, 8])
            elif res == 512:
                trial.suggest_categorical("batch_size", [4, 8, 16])
            else:
                trial.suggest_categorical("batch_size", [8, 16, 32, 64])

            trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            trial.suggest_categorical("encoder_name", ["resnet34", "resnet50", "efficientnet-b0"])
            trial.suggest_float("loss_weight_ratio", 0.0, 1.0)

        # 4. Log the Agent's reasoning before starting the trial
        with get_db_session() as session:
            reasoning_log = AgentReasoningLog(
                trial_id=trial.number,
                study_name=study_name,
                model_version=agent_model,
                prompt_strategy=prompt_strategy,
                predicted_outcome_rationale=reasoning,
                estimated_dice_improvement=estimated_dice_improvement
            )
            session.add(reasoning_log)

        return {
            "success": True,
            "trial_id": trial.number,
            "params": trial.params,
            "message": f"Trial {trial.number} suggested and reasoning logged successfully."
        }
    except Exception as e:
        return {"success": False, "error": f"Error suggesting trial: {str(e)}"}


@mcp.tool()
def report_epoch_performance(
    study_name: str,
    trial_id: int,
    epoch: int,
    dice: float,
    bce: float
) -> Dict[str, Any]:
    """
    Reports intermediate metrics at the end of an epoch.
    Returns whether the trial should be pruned based on Median / ASHA rules.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        # Find the active trial using the trial number/id
        trial = None
        for t in study.trials:
            if t.number == trial_id:
                trial = t
                break
        
        if trial is None:
            return {"success": False, "error": f"Trial {trial_id} not found."}
        
        # In Optuna, to check pruning we report a single metric to the trial object.
        # If it's multi-objective, we construct a composite metric: dice - bce
        # A higher value is better (maximize).
        composite_score = dice - bce
        
        # For Optuna's internal pruner, we check if the study has a single direction or multi.
        # If multi-objective, trial.should_prune() is not natively supported directly, so we run a custom pruning logic:
        # Pruning Rule: If at epoch >= 3, and the composite score is in the bottom 50% of all previous successful trials at that epoch, prune.
        should_prune = False
        
        if len(study.directions) == 1:
            # Single objective (maximizing Dice score)
            # We report to Optuna's trial
            # In TPE ask/tell mode, we must load the trial state as active or report via a study-attached trial
            # Note: Optuna's ask() returns a Trial object. To report intermediate values in ask/tell:
            # We use study._storage.set_trial_intermediate_value(trial._trial_id, step=epoch, value=dice)
            study._storage.set_trial_intermediate_value(trial._trial_id, epoch, dice)
            # To evaluate pruning, we use study.pruner.prune(study, study._storage.get_trial(trial._trial_id))
            should_prune = study.pruner.prune(study, study._storage.get_trial(trial._trial_id))
        else:
            # Multi-objective (Custom Median Pruner on composite score)
            if epoch >= 3:
                # Get historical trials at the exact same epoch
                past_scores = []
                for t in study.trials:
                    if t.number != trial_id and t.state in [TrialState.COMPLETE, TrialState.RUNNING]:
                        intermediate_vals = t.intermediate_values
                        if epoch in intermediate_vals:
                            past_scores.append(intermediate_vals[epoch])
                
                # Report composite score for history tracking
                study._storage.set_trial_intermediate_value(trial._trial_id, epoch, composite_score)
                
                if len(past_scores) >= 3:  # Need at least 3 trials to establish median
                    past_scores.sort()
                    median_val = past_scores[len(past_scores) // 2]
                    if composite_score < median_val:
                        should_prune = True
            else:
                study._storage.set_trial_intermediate_value(trial._trial_id, epoch, composite_score)

        return {
            "success": True,
            "trial_id": trial_id,
            "epoch": epoch,
            "should_prune": should_prune,
            "composite_score": composite_score
        }
    except Exception as e:
        return {"success": False, "error": f"Error reporting epoch performance: {str(e)}"}


@mcp.tool()
def complete_trial(
    study_name: str,
    trial_id: int,
    final_dice: float,
    final_bce: float,
    epoch_reached: int,
    val_loss_history: List[Dict[str, Any]],
    weights_path: str
) -> Dict[str, Any]:
    """
    Marks the trial as complete, registers final deep learning metrics, and
    calculates actual performance improvement relative to the best trial so far.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        trial = None
        for t in study.trials:
            if t.number == trial_id:
                trial = t
                break
        
        if trial is None:
            return {"success": False, "error": f"Trial {trial_id} not found."}
        
        # 1. Tell Optuna the trial results
        # If multi-objective: [BCE, Dice]
        # If single-objective: [Dice]
        if len(study.directions) > 1:
            study.tell(trial.number, [final_bce, final_dice])
        else:
            study.tell(trial.number, final_dice)

        # 2. Write custom deep learning metrics
        with get_db_session() as session:
            metric = SegmentationMetric(
                trial_id=trial_id,
                epoch_reached=epoch_reached,
                final_bce_loss=final_bce,
                final_dice_score=final_dice,
                weights_path=weights_path
            )
            metric.set_history(val_loss_history)
            session.add(metric)

            # 3. Calculate Reasoning Performance Delta
            # Query the best prior Dice score in the study
            prior_trials = [t for t in study.trials if t.number < trial_id and t.state == TrialState.COMPLETE]
            
            best_prior_dice = 0.0
            if prior_trials:
                if len(study.directions) > 1:
                    # In multi-objective, values[1] is Dice Score
                    best_prior_dice = max([t.values[1] for t in prior_trials if t.values and len(t.values) > 1] or [0.0])
                else:
                    best_prior_dice = max([t.value for t in prior_trials if t.value is not None] or [0.0])
            
            actual_improvement = final_dice - best_prior_dice
            
            # Update agent reasoning log
            reasoning_log = session.query(AgentReasoningLog).filter_by(trial_id=trial_id).first()
            if reasoning_log:
                reasoning_log.actual_dice_improvement = actual_improvement
        
        return {
            "success": True,
            "trial_id": trial_id,
            "actual_improvement": actual_improvement,
            "message": f"Trial {trial_id} completed successfully and database records updated."
        }
    except Exception as e:
        return {"success": False, "error": f"Error completing trial: {str(e)}"}


@mcp.tool()
def get_parameter_importance(study_name: str, target_metric: str = "dice") -> Dict[str, Any]:
    """
    Computes parameter importance via functional ANOVA (fANOVA).
    Specifies how much each hyperparameter influences either BCE loss or Dice Score.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) < 2:
            return {"success": False, "message": "Need at least 2 completed trials to compute fANOVA parameter importance."}

        # For target functions in multi-objective:
        # If target_metric is 'bce' (minimize), use t.values[0]
        # If target_metric is 'dice' (maximize), use t.values[1]
        target_fn = None
        if len(study.directions) > 1:
            if target_metric.lower() == "bce":
                target_fn = lambda t: t.values[0]
            else:
                target_fn = lambda t: t.values[1]
        
        importances = get_param_importances(
            study,
            evaluator=FanovaImportanceEvaluator(),
            target=target_fn
        )
        
        # Sort and clean up output
        sorted_importance = {k: float(v) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
        
        return {
            "success": True,
            "study_name": study_name,
            "target_metric": target_metric,
            "parameter_importances": sorted_importance
        }
    except Exception as e:
        return {"success": False, "error": f"Error calculating parameter importance: {str(e)}"}


@mcp.tool()
def get_pareto_front(study_name: str) -> Dict[str, Any]:
    """
    Retrieves the Pareto front for a multi-objective study.
    Returns non-dominated trials showcasing trade-offs between BCE loss (lower is better)
    and Dice Score (higher is better).
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        if len(study.directions) < 2:
            return {
                "success": False,
                "message": "Study is single-objective. Pareto front is only applicable to multi-objective studies."
            }
            
        best_trials = study.best_trials
        pareto_trials = []
        for t in best_trials:
            pareto_trials.append({
                "trial_id": t.number,
                "values": {
                    "bce_loss": t.values[0],
                    "dice_score": t.values[1]
                },
                "params": t.params
            })
            
        return {
            "success": True,
            "study_name": study_name,
            "pareto_front_size": len(pareto_trials),
            "trials": pareto_trials
        }
    except Exception as e:
        return {"success": False, "error": f"Error fetching Pareto front: {str(e)}"}


@mcp.tool()
def get_ai_performance_metrics(study_name: str) -> Dict[str, Any]:
    """
    Evaluates and profiles the AI's hyperparameter search and reasoning capability.
    Returns token/convergence speeds, reasoning alignment MAE, and invalid proposal rates.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        
        with get_db_session() as session:
            # 1. Convergence Speed (trials to Dice > 0.85)
            # Find the trial number with Dice > 0.85
            target_dice = 0.85
            trials_to_converge = None
            
            # Sort trials chronologically
            completed_trials = sorted(
                [t for t in study.trials if t.state == TrialState.COMPLETE],
                key=lambda t: t.number
            )
            
            for index, t in enumerate(completed_trials):
                # Retrieve dice from values or database
                dice_val = 0.0
                if len(study.directions) > 1:
                    dice_val = t.values[1] if t.values and len(t.values) > 1 else 0.0
                else:
                    dice_val = t.value if t.value is not None else 0.0
                
                if dice_val >= target_dice:
                    trials_to_converge = index + 1
                    break
            
            # 2. Invalid Proposal Rate (hallucinations/guardrails)
            invalid_count = session.query(InvalidProposal).filter_by(study_name=study_name).count()
            total_proposals = len(study.trials) + invalid_count
            invalid_rate = (invalid_count / total_proposals) if total_proposals > 0 else 0.0
            
            # 3. Reasoning Alignment Profiling
            # Query reasoning logs that have actual outcomes populated
            logs = session.query(AgentReasoningLog).filter(
                AgentReasoningLog.study_name == study_name,
                AgentReasoningLog.actual_dice_improvement.isnot(None)
            ).all()
            
            alignment_data = []
            mae_delta = 0.0
            total_valid_logs = len(logs)
            
            for log in logs:
                est = log.estimated_dice_improvement
                act = log.actual_dice_improvement
                delta = abs(est - act)
                mae_delta += delta
                alignment_data.append({
                    "trial_id": log.trial_id,
                    "model_version": log.model_version,
                    "prompt_strategy": log.prompt_strategy,
                    "estimated": est,
                    "actual": act,
                    "absolute_error": delta
                })
            
            mean_absolute_error = (mae_delta / total_valid_logs) if total_valid_logs > 0 else 0.0
            
            # Aggregate stats per model configuration
            model_stats = {}
            for d in alignment_data:
                m = d["model_version"]
                if m not in model_stats:
                    model_stats[m] = {"errors": [], "count": 0}
                model_stats[m]["errors"].append(d["absolute_error"])
                model_stats[m]["count"] += 1
                
            model_breakdown = []
            for model_name, info in model_stats.items():
                model_breakdown.append({
                    "model_version": model_name,
                    "evaluations_count": info["count"],
                    "mean_absolute_error": sum(info["errors"]) / len(info["errors"])
                })

            return {
                "success": True,
                "study_name": study_name,
                "convergence": {
                    "target_dice_score": target_dice,
                    "trials_to_reach_target": trials_to_converge,
                    "total_completed_trials": len(completed_trials)
                },
                "guardrails": {
                    "total_invalid_proposals": invalid_count,
                    "total_valid_proposals": len(study.trials),
                    "invalid_proposal_rate": invalid_rate
                },
                "reasoning_accuracy": {
                    "evaluated_trials": total_valid_logs,
                    "overall_mean_absolute_error": mean_absolute_error,
                    "model_performance_breakdown": model_breakdown,
                    "history": alignment_data
                }
            }
    except Exception as e:
        return {"success": False, "error": f"Error compiling AI metrics: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
