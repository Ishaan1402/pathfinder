import os
import json
import datetime
import hashlib
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import requests
import optuna
from optuna.trial import TrialState
from mcp.server.fastmcp import FastMCP

from src.db_manager import init_db, get_db_session, DATABASE_URL
from src.schema import (
    TrialResult,
    TrialMetadata,
    SystemConfiguration,
    CompactedPacket,
    StudyCard,
    AgentReasoningLog,
    StudyReview,
    StudyStatus
)
from src.hpo_config import load_hpo_config, normalize_trial_params
from src.analytics import build_compacted_packet
from src.hpo_coordinator import (
    compute_health_tier,
    count_evaluated_trials,
    POLICY_ACTIONS,
    build_review_packet,
    load_study_cards
)

from src.search_space import (
    load_search_space,
    save_search_space,
    _apply_search_space_patch,
)
from src.suggest import get_or_create_study, _enqueue_manual_trial

# Initialize database schema and run migrations on startup
init_db()

# Create FastMCP server
mcp = FastMCP("Pathfinder")

from src.hpo_coordinator import _validate_manual_parameters

# --- MCP TOOLS ---

from src.onboarding import (
    initialize_study as core_initialize_study,
    delete_study_internal,
    init_study_from_manifest_dict
)

@mcp.tool()
def initialize_study(
    study_name: str,
    active_search_space: Dict[str, Any],
    hpo_config: Dict[str, Any],
    project_context: Optional[Dict[str, Any]] = None,
    source_files: Optional[Dict[str, str]] = None,
    multi_objective: bool = True,
    directions: Optional[List[str]] = None
) -> str:
    """Initializes a new study: creates Optuna study and stores search space, config, context, and source files in DB."""
    return core_initialize_study(
        study_name=study_name,
        active_search_space=active_search_space,
        hpo_config=hpo_config,
        project_context=project_context,
        source_files=source_files, multi_objective=multi_objective,
        directions=directions
    )

@mcp.tool()
def get_study_data(study_name: str) -> Dict[str, Any]:
    """Returns the compacted HPO review packet, utilizing a lazy materialization cache layer."""
    return build_review_packet(study_name)

@mcp.tool()
def validate_search_space(
    space_config: Dict[str, Any],
    hpo_config: Optional[Dict[str, Any]] = None,
    project_context: Optional[Dict[str, Any]] = None,
    historical_fail_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate search space bounds, tunable coverage, and metric label consistency."""
    errors = []
    warnings = []
    tunable_count = 0

    for param_name, spec in space_config.items():
        if param_name.startswith("_") or not isinstance(spec, dict):
            continue
        ptype = spec.get("type")
        if ptype in ("float", "float_log", "int"):
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is None or hi is None:
                errors.append(f"Parameter '{param_name}' is missing required min/max bounds.")
            else:
                try:
                    lo_f = float(lo)
                    hi_f = float(hi)
                    if lo_f >= hi_f:
                        errors.append(f"Parameter '{param_name}': min ({lo}) must be strictly less than max ({hi}).")
                    if ptype == "float_log" and lo_f <= 0:
                        errors.append(f"Parameter '{param_name}' is log-scale and must have a min bound strictly greater than 0.")
                    if ptype == "float_log" and hi_f > 0 and lo_f > 0:
                        import math
                        span_orders = math.log10(hi_f) - math.log10(lo_f)
                        if span_orders > 6:
                            warnings.append(
                                f"Parameter '{param_name}' log-span spans {span_orders:.1f} orders of magnitude (>6); consider narrowing."
                            )
                    if hi_f > lo_f:
                        tunable_count += 1
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param_name}' has non-numeric min/max bounds.")
        elif ptype == "categorical":
            options = spec.get("options", [])
            active = spec.get("active", options)
            if not options:
                errors.append(f"Categorical parameter '{param_name}' must specify allowed options.")
            elif len(active) == 0:
                errors.append(f"Categorical parameter '{param_name}' must have at least one active option.")
            elif len(active) == 1:
                warnings.append(f"Categorical parameter '{param_name}' has only 1 active choice ({active[0]}), effectively pinning it.")
            elif len(active) > 1:
                tunable_count += 1
            
            # Check active in options
            invalid_active = [x for x in active if x not in options]
            if invalid_active:
                errors.append(f"Categorical parameter '{param_name}' active options {invalid_active} are not in choices: {options}")

    # Check known OOM-risk combinations
    batch_size_spec = space_config.get("batch_size", {})
    resolution_spec = space_config.get("resolution", {})
    
    max_bs = None
    if batch_size_spec.get("type") == "categorical":
        max_bs = max(batch_size_spec.get("active", [0]))
    elif batch_size_spec.get("type") in ("int", "float"):
        max_bs = batch_size_spec.get("max")

    max_res = None
    if resolution_spec.get("type") == "categorical":
        max_res = max(resolution_spec.get("active", [0]))
    elif resolution_spec.get("type") in ("int", "float"):
        max_res = resolution_spec.get("max")

    if max_bs is not None and max_res is not None:
        try:
            if float(max_bs) >= 64 and float(max_res) >= 1024:
                warnings.append(f"High risk configuration: batch_size={max_bs} combined with resolution={max_res} has historically high OOM risk.")
        except Exception:
            pass

    if tunable_count == 0:
        errors.append("Search space has no tunable parameters (all bounds pinned or single-choice categoricals).")

    if project_context:
        ctx = project_context if isinstance(project_context, dict) else {}
        declared_score = ctx.get("metric_score_name") or ctx.get("score_metric")
        declared_loss = ctx.get("metric_loss_name") or ctx.get("loss_metric")
        if hpo_config and (declared_score or declared_loss):
            if declared_score and not hpo_config.get("metric_score_label"):
                warnings.append("project_context declares a score metric but hpo_config.metric_score_label is missing.")
            if declared_loss and not hpo_config.get("metric_loss_label"):
                warnings.append("project_context declares a loss metric but hpo_config.metric_loss_label is missing.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

@mcp.tool()
def update_search_space(study_name: str, space_config: Dict[str, Any], apply: bool = False) -> str:
    """Propose or apply updates to the active search space.

    Every change is validated against the canonical search space (legacy parameter aliases
    already normalized). With apply=False the change is staged as ``pending_search_space`` for
    the human to approve on the dashboard; apply=True commits immediately. Unrecognized
    parameters, out-of-bounds categorical choices, and no-op proposals return an explicit error
    string instead of being silently dropped.
    """
    current_space = load_search_space(study_name)
    validated_proposals: Dict[str, Any] = {}

    for param_name, new_val in space_config.items():
        if param_name not in current_space:
            return f"Error: Hyperparameter '{param_name}' is not recognized in the search space."
        param_type = current_space[param_name].get("type")
        proposal: Dict[str, Any] = {}
        if param_type == "categorical":
            if "active" in new_val:
                allowed = current_space[param_name].get("options", [])
                invalid_options = [x for x in new_val["active"] if x not in allowed]
                if invalid_options:
                    return f"Error: Active choices {invalid_options} for {param_name} are not in options: {allowed}"
                if len(new_val["active"]) == 0:
                    return f"Error: Categorical parameter {param_name} must have at least one active option."
                proposal["active"] = new_val["active"]
        else:
            if "min" in new_val:
                proposal["min"] = float(new_val["min"])
            if "max" in new_val:
                proposal["max"] = float(new_val["max"])
        if not proposal:
            return (
                f"Error: No valid changes for '{param_name}'. Provide 'active' for categorical "
                f"parameters, or 'min'/'max' for numeric parameters."
            )
        validated_proposals[param_name] = proposal

    if apply:
        for key, new_val in validated_proposals.items():
            current_space[key].update(new_val)
        save_search_space(current_space, study_name)
        with get_db_session() as session:
            pending = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="pending_search_space"
            ).first()
            if pending:
                session.delete(pending)
        from src.hpo_coordinator import mark_review_applied
        mark_review_applied(study_name)
        return "Search space changes committed successfully."

    with get_db_session() as session:
        session.merge(SystemConfiguration(
            study_name=study_name,
            config_key="pending_search_space",
            config_value=json.dumps(validated_proposals),
        ))
    return "Search space changes proposed successfully. They must be approved via the dashboard before they take effect."

@mcp.tool()
def delete_study(study_name: str, confirm: bool = False) -> Dict[str, Any]:
    """Permanently delete a study: its Optuna trials and ALL custom metadata rows."""
    return delete_study_internal(study_name=study_name, confirm=confirm)


@mcp.tool()
def generate_model_card(study_name: str) -> Dict[str, Any]:
    """Generates an end-of-study synthesis, writes MODEL_CARD.md to disk, and indexes it in DB."""
    try:
        # Load data
        packet = get_study_data(study_name)
        if "error" in packet:
            return packet

        best_params = {}
        best_score = 0.0
        elite = packet.get("trial_bins", {}).get("elite", [])
        if elite:
            best_params = elite[0].get("params", {})
            best_score = elite[0].get("primary_score", 0.0)

        # Construct beautiful model card Markdown
        card_content = f"""# Study Model Card: {study_name}

## Executive Summary
This model card synthesizes results for study `{study_name}`.

- **Best Achieved Score ({packet.get('metric_score_label', 'Score')}):** {best_score:.4f}
- **Optimal Hyperparameters:**
{chr(10).join(f"  - `{k}`: {v}" for k, v in best_params.items())}

## Search Space Performance
- **Total Trials Evaluated:** {packet.get('counts', {}).get('total', 0)}
- **Successful Runs:** {packet.get('counts', {}).get('complete', 0)}
- **Pruned Runs:** {packet.get('counts', {}).get('pruned', 0)}
- **Failed/OOM Runs:** {packet.get('counts', {}).get('failed', 0)}

### Key Parameter Importances (fANOVA)
{chr(10).join(f"- `{k}`: {v:.4f}" for k, v in packet.get('fanova_importances', {}).items())}

## Telemetry Profile
- **GPU Device:** {packet.get('vram_telemetry', {}).get('gpu_model', 'Unknown')}
- **Peak VRAM Recorded:** {packet.get('vram_telemetry', {}).get('gpu_capacity_gb', 0.0):.2f} GB
- **OOM Failures:** {packet.get('vram_telemetry', {}).get('oom_count', 0)}

---
*Generated by Pathfinder on {datetime.datetime.utcnow().isoformat()}*
"""

        # Write to studies directory in workspace
        studies_dir = os.path.join(os.path.dirname(__file__), "studies")
        os.makedirs(studies_dir, exist_ok=True)
        
        file_path = os.path.join(studies_dir, f"{study_name}_model_card.md")
        with open(file_path, "w") as f:
            f.write(card_content)

        # Hash calculation
        sha = hashlib.sha256(card_content.encode("utf-8")).hexdigest()

        # Save card index to database
        with get_db_session() as session:
            session.merge(StudyCard(
                study_name=study_name,
                card_type="model_card",
                file_path=os.path.relpath(file_path, os.path.dirname(__file__)),
                content_hash=sha,
                metadata_json=json.dumps({
                    "best_score": best_score,
                    "best_params": best_params,
                    "total_trials": packet.get("counts", {}).get("total", 0)
                })
            ))

        return {
            "success": True,
            "file_path": file_path,
            "content_hash": sha,
            "message": f"Model card written to disk and database index updated."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to generate model card: {str(e)}"}

@mcp.tool()
def submit_agent_review(
    study_name: str,
    summary: str,
    health_rating: int,
    policy_action: str = "no_change",
    model_version: str = "coordinator",
    prompt_strategy: str = "coordinator_review",
    reasons: Optional[List[Dict[str, Any]]] = None,
    estimated_score_improvement: Optional[float] = None,
    cited_best_trial: Optional[int] = None,
    search_space_patch: Optional[Dict[str, Any]] = None,
    manual_trial: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Persists a coordinator review. Idempotent per trial window. Matches the HTTP route logic."""
    try:
        from src.hpo_coordinator import (
            save_study_review,
            count_evaluated_trials,
            POLICY_ACTIONS,
            validate_review_fields,
        )

        if policy_action not in POLICY_ACTIONS:
            return {
                "success": False,
                "error": f"Invalid policy_action '{policy_action}'. Valid options are: {', '.join(POLICY_ACTIONS)}"
            }

        study = get_or_create_study(study_name)
        space = load_search_space(study_name)
        trials_evaluated = count_evaluated_trials(study)

        if manual_trial:
            val_res = _validate_manual_parameters(manual_trial, study_name)
            if not val_res["ok"]:
                with get_db_session() as session:
                    from src.schema import InvalidProposal
                    session.add(InvalidProposal(
                        study_name=study_name,
                        model_version=model_version or "coordinator",
                        prompt_strategy=prompt_strategy or "coordinator_review",
                        invalid_parameters=json.dumps(manual_trial),
                        validation_error=val_res["error"]
                    ))
                return {"success": False, "error": f"Invalid manual parameters: {val_res['error']}"}

        validation = validate_review_fields(estimated_score_improvement, cited_best_trial)
        if not validation["ok"]:
            return {"success": False, "error": "; ".join(validation["errors"])}

        result = save_study_review(
            study_name,
            summary,
            health_rating=health_rating,
            policy_action=policy_action or "no_change",
            model_version=model_version or "coordinator",
            prompt_strategy=prompt_strategy or "coordinator_review",
            reasons=reasons,
            trials_evaluated=trials_evaluated,
            estimated_score_improvement=estimated_score_improvement,
            cited_best_trial=cited_best_trial,
            force=force,
        )

        applied = {}
        if not result.get("duplicate"):
            if search_space_patch:
                applied["search_space"] = _apply_search_space_patch(search_space_patch, space, study_name)
            if manual_trial:
                applied["manual_trial"] = _enqueue_manual_trial(study, manual_trial, space, summary)

        result["applied"] = applied
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@mcp.tool()
def validate_integration(study_name: str) -> Dict[str, Any]:
    """Validates that a study is correctly initialized and configured in SQLite."""
    try:
        from src.db_manager import get_db_session
        from src.schema import SystemConfiguration, StudyStatus
        import optuna
        
        status = {}
        try:
            study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
            status["optuna_study_exists"] = True
            status["study_directions"] = [d.name for d in study.directions]
            status["total_trials"] = len(study.trials)
        except Exception as e:
            status["optuna_study_exists"] = False
            status["optuna_study_error"] = str(e)
            
        with get_db_session() as session:
            space = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="active_search_space"
            ).first()
            config = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="hpo_config"
            ).first()
            status["db_search_space_configured"] = space is not None
            status["db_hpo_config_configured"] = config is not None
            
            status_row = session.query(StudyStatus).filter_by(study_name=study_name).first()
            if status_row:
                status["health_tier"] = status_row.health_tier
                status["health_reason"] = status_row.health_reason
            else:
                status["health_tier"] = "unknown"
                
        broker_url = os.getenv("HPO_BROKER_URL", "http://localhost:8000")
        status["broker_url"] = broker_url
        try:
            resp = requests.get(f"{broker_url.rstrip('/')}/health", timeout=3)
            status["broker_online"] = resp.status_code == 200
        except Exception as e:
            status["broker_online"] = False
            status["broker_error"] = str(e)
            
        status["success"] = status.get("optuna_study_exists", False) and status.get("db_search_space_configured", False)
        return status
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_study_cards(study_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves generated study cards (model cards, recaps) from the database to enable cross-study queries."""
    return load_study_cards(study_name)

@mcp.tool()
def validate_manifest(yaml_str: str) -> Dict[str, Any]:
    """Mechanically validate a manifest YAML string against the Pathfinder schema rules."""
    import yaml
    from src.manifest import validate_manifest as core_validate
    try:
        data = yaml.safe_load(yaml_str)
    except Exception as e:
        return {"success": False, "errors": [f"Invalid YAML structure: {str(e)}"], "warnings": []}
        
    if not isinstance(data, dict):
        return {"success": False, "errors": ["Manifest root must be a dictionary"], "warnings": []}

    errors, warnings = core_validate(data)
    return {"success": len(errors) == 0, "errors": errors, "warnings": warnings}

@mcp.tool()
def init_from_manifest(yaml_str: str, force: bool = False) -> Dict[str, Any]:
    """Validate and register a new HPO study from a manifest YAML string, with deep overwrite cleanup on force=True."""
    import yaml
    try:
        data = yaml.safe_load(yaml_str)
    except Exception as e:
        return {"success": False, "error": f"Invalid YAML structure: {str(e)}"}
        
    if not isinstance(data, dict):
        return {"success": False, "error": "Manifest root must be a dictionary"}

    try:
        result = init_study_from_manifest_dict(data, force=force)
        return {"success": True, "message": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def export_manifest(study_name: str) -> str:
    """Export the active search space, HPO config, and context of an existing study as a valid manifest YAML string."""
    import yaml
    import json
    
    # Query database configurations
    space_val = None
    config_val = None
    context_val = None
    source_val = None

    with get_db_session() as session:
        space_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        if space_row:
            space_val = space_row.config_value
            
        config_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="hpo_config"
        ).first()
        if config_row:
            config_val = config_row.config_value
            
        context_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="project_context"
        ).first()
        if context_row:
            context_val = context_row.config_value
            
        source_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="source_files"
        ).first()
        if source_row:
            source_val = source_row.config_value

    if not space_val:
        raise ValueError(f"Study '{study_name}' not found in database.")

    space = json.loads(space_val)
    hpo_config = json.loads(config_val) if config_val else {}
    project_context = json.loads(context_val) if context_val else {}
    source_files = json.loads(source_val) if source_val else {}

    # Reconstruct manifest dict
    manifest = {
        "study_name": study_name
    }

    if "manifest_metrics" in hpo_config:
        manifest["metrics"] = hpo_config["manifest_metrics"]
    else:
        score_label = hpo_config.get("metric_score_label", "score")
        loss_label = hpo_config.get("metric_loss_label", "loss")
        
        try:
            study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
            directions = [d.name.lower() for d in study.directions]
        except Exception:
            directions = ["maximize"]  # Default fallback

        # Reconstruct metrics objectives
        objectives = []
        primary_score = "score" # default fallback
        
        if len(directions) > 1:
            objectives.append({
                "name": "loss",
                "direction": "minimize",
                "label": loss_label
            })
            objectives.append({
                "name": "score",
                "direction": "maximize",
                "label": score_label
            })
            primary_score = "score"
        else:
            dir_name = "maximize"
            if directions:
                dir_name = "minimize" if directions[0] == "minimize" else "maximize"
            objectives.append({
                "name": "score",
                "direction": dir_name,
                "label": score_label
            })
            primary_score = "score"

        manifest["metrics"] = {
            "primary_score": primary_score,
            "objectives": objectives
        }

    # Reconstruct params
    params_list = []
    for p_name, spec in space.items():
        if not isinstance(spec, dict):
            continue
        ptype = spec.get("type")
        param_def = {"name": p_name}
        
        if ptype in ("float", "float_log", "int"):
            param_def["type"] = ptype
            param_def["min"] = spec.get("min")
            param_def["max"] = spec.get("max")
        elif ptype == "categorical":
            opts = spec.get("options", [])
            if len(opts) == 1:
                param_def["type"] = "fixed"
                param_def["value"] = opts[0]
            elif set(opts) == {True, False}:
                param_def["type"] = "bool"
            else:
                param_def["type"] = "categorical"
                param_def["options"] = opts
        else:
            param_def["type"] = "fixed"
            param_def["value"] = spec.get("value")
            
        params_list.append(param_def)

    manifest["params"] = params_list

    # Eval protocol
    eval_proto = hpo_config.get("eval_protocol", {})
    if eval_proto and eval_proto.get("enabled"):
        manifest["eval_protocol"] = {
            "enabled": True,
            "fixed_resolution": eval_proto.get("fixed_resolution"),
            "train_resolution_param": eval_proto.get("train_resolution_param", "resolution")
        }

    # Worker entrypoint/env from project_context
    worker_data = {}
    if "worker_entrypoint" in project_context:
        worker_data["entrypoint"] = project_context["worker_entrypoint"]
    if "worker_env" in project_context:
        worker_data["env"] = project_context["worker_env"]
    if worker_data:
        manifest["worker"] = worker_data

    # Exclude system metadata from project_context
    filtered_context = {k: v for k, v in project_context.items() if k not in ("worker_entrypoint", "worker_env")}
    if filtered_context:
        manifest["project_context"] = filtered_context

    if source_files:
        manifest["source_files"] = source_files

    return yaml.dump(manifest, sort_keys=False)

# --- MCP PROMPT RESOURCES ---

@mcp.resource("hpo://prompts/grill")
def resource_grill() -> str:
    """Onboarding checklist: interview, then manifest loop."""
    return """# Pathfinder Onboarding (Grill + Manifest Loop)

See AGENTS.md for the full procedure. After interviewing the user (metrics, GPU, bounds, hypothesis):

1. Draft a YAML manifest configuration (e.g. `train.hpo.yaml`).
2. Call `validate_manifest(yaml_str)` to check for errors/warnings mechanically.
3. Call `init_from_manifest(yaml_str)` to register the study in SQLite and Optuna.
4. Call `validate_integration(study_name)` to confirm the broker is healthy and integration is ready.

Worker integration reference: `docs/INTEGRATION.md`. Do not write json space config files to disk.
"""

@mcp.resource("hpo://prompts/review")
def resource_review() -> str:
    """7-step episodic coordinator review (human-initiated only)."""
    return """# Pathfinder Coordinator Review (7 Steps)

Follow AGENTS.md. Trigger only when the user explicitly requests a review (watch/intervene nudges are not automatic).

1. `get_study_data(study_name)` — packet includes fANOVA, past_reviews, coordinator_accuracy, statistical_confidence.
2. Interpret metrics using dynamic labels from project_context; heed statistical_confidence caveat when low/medium.
3. VRAM safety via vram_telemetry bounds_oom_risk; check past_reviews (ignore quality_flagged).
4. Self-regulate only if coordinator_accuracy.n_scored_reviews >= 3 and MAE > 0.05.
5. `update_search_space(..., apply=False)` to stage bounds (human approves on dashboard).
6. `submit_agent_review` with required estimated_score_improvement and cited_best_trial.
7. `generate_model_card(study_name)` when wrapping up.
"""

if __name__ == "__main__":
    mcp.run()
