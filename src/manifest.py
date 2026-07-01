# src/manifest.py — manifest schema definition and validation

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class ParamType(str, Enum):
    FLOAT = "float"
    FLOAT_LOG = "float_log"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"
    FIXED = "fixed"

class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass
class ObjectiveDef:
    name: str                          # e.g. "roc_auc", "loss"
    direction: ObjectiveDirection
    label: str                         # e.g. "ROC AUC", "Cross-Entropy Loss"

@dataclass  
class ParamDef:
    name: str
    type: ParamType
    min: Optional[float] = None        # for float/float_log/int
    max: Optional[float] = None        # for float/float_log/int
    options: Optional[List[Any]] = None  # for categorical
    value: Optional[Any] = None        # for fixed (constant, not tuned)

@dataclass
class EvalProtocolDef:
    enabled: bool = False
    fixed_resolution: Optional[int] = None
    train_resolution_param: str = "resolution"
    score_eval_attr: str = "score_eval_fixed"   # generically named now
    loss_eval_attr: str = "loss_eval_fixed"

@dataclass
class WorkerDef:
    entrypoint: str = ""               # e.g. "python train.py"
    env: Dict[str, str] = field(default_factory=dict)

@dataclass
class ManifestDef:
    study_name: str
    metrics: 'MetricsDef'
    params: List[ParamDef]
    eval_protocol: EvalProtocolDef = field(default_factory=EvalProtocolDef)
    worker: WorkerDef = field(default_factory=WorkerDef)
    project_context: Dict[str, Any] = field(default_factory=dict)
    source_files: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricsDef:
    primary_score: str                 # name of the objective to display/highlight
    objectives: List[ObjectiveDef]


def get_distinct_style(name: str) -> Optional[str]:
    if "_" in name:
        return "snake_case"
    if "-" in name:
        return "kebab-case"
    if any(c.isupper() for c in name):
        if name[0].isupper():
            return "PascalCase"
        else:
            return "camelCase"
    return None


def validate_manifest(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Validate raw manifest data against Pathfinder schema rules."""
    errors = []
    warnings = []

    if not isinstance(data, dict):
        return ["Manifest root must be a dictionary"], []

    # 1. study_name is present and non-empty
    study_name = data.get("study_name")
    if not study_name or not isinstance(study_name, str) or not study_name.strip():
        errors.append("study_name is missing or empty")
    
    # Check metrics
    metrics = data.get("metrics")
    objectives = []
    primary_score = None
    if not metrics or not isinstance(metrics, dict):
        errors.append("metrics section is missing or invalid")
    else:
        primary_score = metrics.get("primary_score")
        objectives = metrics.get("objectives")

        # 2. metrics.objectives has at least one objective
        if not objectives or not isinstance(objectives, list):
            errors.append("metrics.objectives must contain at least one objective definition")
        else:
            # Rule 15: Support at most 2 objectives (one maximize, one minimize)
            if len(objectives) > 2:
                errors.append("At most 2 objectives can be defined in metrics.objectives")
            elif len(objectives) == 2:
                dirs = [obj.get("direction") for obj in objectives if isinstance(obj, dict)]
                if "maximize" not in dirs or "minimize" not in dirs:
                    errors.append("When defining 2 objectives, one must be 'maximize' and the other must be 'minimize'")

            obj_names = set()
            for i, obj in enumerate(objectives):
                if not isinstance(obj, dict):
                    errors.append(f"Objective at index {i} must be a dictionary")
                    continue
                name = obj.get("name")
                direction = obj.get("direction")
                label = obj.get("label")

                # 4. Every objective has: name (non-empty), direction ("maximize" or "minimize"), label (non-empty)
                if not name or not isinstance(name, str) or not name.strip():
                    errors.append(f"Objective at index {i} is missing a valid name")
                else:
                    # 5. No duplicate objective names
                    if name in obj_names:
                        errors.append(f"Duplicate objective name found: '{name}'")
                    obj_names.add(name)

                if direction not in ("maximize", "minimize"):
                    errors.append(f"Objective '{name or i}' direction must be 'maximize' or 'minimize'")
                
                if not label or not isinstance(label, str) or not label.strip():
                    errors.append(f"Objective '{name or i}' is missing a valid label")

            # 3. metrics.primary_score references an objective name that exists
            if primary_score not in obj_names:
                errors.append("metrics.primary_score must reference a valid defined objective name")

            # Warning 5: Single objective with "minimize" direction
            if len(objectives) == 1 and objectives[0].get("direction") == "minimize" if isinstance(objectives[0], dict) else False:
                warnings.append("Only objective is minimize — are you sure the primary_score shouldn't be a derived metric?")

    # Check params
    params = data.get("params")
    reserved_names = {"trial_id", "study_name", "worker_id", "epoch", "state"}
    if not params or not isinstance(params, list):
        errors.append("params must contain at least one tunable parameter (non-fixed type)")
    else:
        # 6. params has at least one tunable parameter
        tunable_count = 0
        param_names = set()
        param_styles = set()

        for i, param in enumerate(params):
            if not isinstance(param, dict):
                errors.append(f"Parameter at index {i} must be a dictionary")
                continue
            name = param.get("name")
            ptype = param.get("type")

            # 7. Every param has: name (non-empty), type (one of the 6 valid types)
            if not name or not isinstance(name, str) or not name.strip():
                errors.append(f"Parameter at index {i} is missing a valid name")
                continue

            # 13. No duplicate param names
            if name in param_names:
                errors.append(f"Duplicate parameter name found: '{name}'")
            param_names.add(name)

            # 14. No param names that conflict with Pathfinder reserved names
            if name in reserved_names:
                errors.append(f"Parameter '{name}' conflicts with Pathfinder reserved names")

            # Style tracking
            style = get_distinct_style(name)
            if style:
                param_styles.add(style)

            valid_types = [t.value for t in ParamType]
            if ptype not in valid_types:
                errors.append(f"Parameter '{name}' has invalid type '{ptype}'. Must be one of {valid_types}")
                continue

            if ptype != ParamType.FIXED:
                tunable_count += 1

            # 8. For float/float_log/int params: min and max are present, min < max
            if ptype in (ParamType.FLOAT, ParamType.FLOAT_LOG, ParamType.INT):
                p_min = param.get("min")
                p_max = param.get("max")

                if p_min is None or p_max is None:
                    errors.append(f"Parameter '{name}' must specify 'min' and 'max' bounds")
                else:
                    try:
                        p_min = float(p_min)
                        p_max = float(p_max)
                        if p_min >= p_max:
                            errors.append(f"Parameter '{name}' min bound must be strictly less than max bound")
                    except (TypeError, ValueError):
                        errors.append(f"Parameter '{name}' bounds must be numeric values")
                        continue

                    # 9. For float_log params: min > 0
                    if ptype == ParamType.FLOAT_LOG and p_min <= 0:
                        errors.append(f"Parameter '{name}' of type float_log must have min bound > 0")

                    # Warning 1: Narrow bounds
                    if p_max != 0 and (p_max - p_min) < 0.01 * abs(p_max):
                        warnings.append(f"Consider widening bounds for {name}: min and max are nearly identical")

                    # Warning 2: Wide log span
                    if ptype == ParamType.FLOAT_LOG and p_min > 0 and p_max > 0:
                        if math.log10(p_max / p_min) > 6:
                            warnings.append(f"Parameter {name} spans {int(math.log10(p_max / p_min))} orders of magnitude — unusually wide")

            # 10. For categorical params: options is a non-empty list
            elif ptype == ParamType.CATEGORICAL:
                options = param.get("options")
                if not isinstance(options, list) or not options:
                    errors.append(f"Parameter '{name}' of type categorical must have a non-empty options list")
                else:
                    # Warning 3: Single categorical option
                    if len(options) == 1:
                        warnings.append(f"Parameter {name} has only one option — should it be 'fixed'?")

            # 12. For fixed params: value is present
            elif ptype == ParamType.FIXED:
                if "value" not in param:
                    errors.append(f"Parameter '{name}' of type fixed must have a 'value' defined")

        # 6. params has at least one tunable parameter
        if tunable_count == 0:
            errors.append("params must contain at least one tunable parameter (non-fixed type)")

        # Warning 6: More than 10 params
        if len(params) > 10:
            warnings.append("Large search space — consider starting with fewer parameters")

        # Warning 7: params with mixed naming styles
        if len(param_styles) > 1:
            warnings.append(f"Parameter names use inconsistent styles: {sorted(list(param_styles))}")

    # Warning 4: No eval_protocol configured
    eval_protocol = data.get("eval_protocol")
    if not eval_protocol or not isinstance(eval_protocol, dict) or not eval_protocol.get("enabled"):
        warnings.append("Eval protocol is disabled — pruning will use train metrics")

    # Validate validation_rules
    rules = data.get("validation_rules")
    if rules:
        if not isinstance(rules, dict):
            errors.append("validation_rules must be a dictionary")
        else:
            enabled = rules.get("enabled")
            score_min = rules.get("score_min")
            loss_min = rules.get("loss_min")
            max_epoch_jump = rules.get("max_epoch_jump")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append("validation_rules.enabled must be a boolean")
            if score_min is not None and not isinstance(score_min, (int, float)):
                errors.append("validation_rules.score_min must be a numeric value")
            if loss_min is not None and not isinstance(loss_min, (int, float)):
                errors.append("validation_rules.loss_min must be a numeric value")
            if max_epoch_jump is not None and not isinstance(max_epoch_jump, (int, float)):
                errors.append("validation_rules.max_epoch_jump must be a numeric value")

    return errors, warnings


def export_manifest_yaml(study_name: str) -> str:
    """Export the active search space, HPO config, and context of an existing study as a valid manifest YAML string."""
    import yaml
    import json
    import optuna
    from .db_manager import get_db_session, DATABASE_URL
    from .schema import SystemConfiguration

    space_val: Optional[str] = None
    config_val: Optional[str] = None
    context_val: Optional[str] = None
    source_val: Optional[str] = None

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

    manifest = {"study_name": study_name}

    if "manifest_metrics" in hpo_config:
        manifest["metrics"] = hpo_config["manifest_metrics"]
    else:
        score_label = hpo_config.get("metric_score_label", "score")
        loss_label = hpo_config.get("metric_loss_label", "loss")

        try:
            study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)
            directions = [d.name.lower() for d in study.directions]
        except Exception:
            directions = ["maximize"]

        objectives = []
        primary_score = "score"

        if len(directions) > 1:
            objectives.append({"name": "loss", "direction": "minimize", "label": loss_label})
            objectives.append({"name": "score", "direction": "maximize", "label": score_label})
            primary_score = "score"
        else:
            dir_name = "minimize" if directions and directions[0] == "minimize" else "maximize"
            objectives.append({"name": "score", "direction": dir_name, "label": score_label})
            primary_score = "score"

        manifest["metrics"] = {"primary_score": primary_score, "objectives": objectives}

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

    eval_proto = hpo_config.get("eval_protocol", {})
    if eval_proto and eval_proto.get("enabled"):
        manifest["eval_protocol"] = {
            "enabled": True,
            "fixed_resolution": eval_proto.get("fixed_resolution"),
            "train_resolution_param": eval_proto.get("train_resolution_param", "resolution"),
        }

    worker_data = {}
    if "worker_entrypoint" in project_context:
        worker_data["entrypoint"] = project_context["worker_entrypoint"]
    if "worker_env" in project_context:
        worker_data["env"] = project_context["worker_env"]
    if worker_data:
        manifest["worker"] = worker_data

    filtered_context = {k: v for k, v in project_context.items() if k not in ("worker_entrypoint", "worker_env")}
    if filtered_context:
        manifest["project_context"] = filtered_context

    if source_files:
        manifest["source_files"] = source_files

    return yaml.dump(manifest, sort_keys=False)


def _manifest_params_to_search_space(params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map manifest parameter definitions into active_search_space schema dictionary."""
    space = {}
    for p in params:
        name = p["name"]
        ptype = p["type"]
        
        if ptype == ParamType.FLOAT:
            space[name] = {"min": float(p["min"]), "max": float(p["max"]), "type": "float"}
        elif ptype == ParamType.FLOAT_LOG:
            space[name] = {"min": float(p["min"]), "max": float(p["max"]), "type": "float_log"}
        elif ptype == ParamType.INT:
            space[name] = {"min": int(p["min"]), "max": int(p["max"]), "type": "int"}
        elif ptype == ParamType.CATEGORICAL:
            opts = p["options"]
            space[name] = {"options": opts, "active": list(opts), "type": "categorical"}
        elif ptype == ParamType.BOOL:
            space[name] = {"options": [True, False], "active": [True, False], "type": "categorical"}
        elif ptype == ParamType.FIXED:
            space[name] = {"options": [p["value"]], "active": [p["value"]], "type": "categorical"}
            
    return space


def _manifest_to_hpo_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map manifest definition into fallback HPO_config database schema."""
    metrics = data.get("metrics", {})
    primary_score = metrics.get("primary_score", "")
    objectives = metrics.get("objectives", [])
    
    # Determine metric labels
    loss_label = "loss"
    score_label = "score"
    
    for obj in objectives:
        obj_name = obj.get("name")
        obj_label = obj.get("label", obj_name)
        if obj_name == primary_score:
            score_label = obj_label
        elif "loss" in obj_name.lower() or "bce" in obj_name.lower():
            loss_label = obj_label

    eval_proto = data.get("eval_protocol", {})
    eval_proto_enabled = bool(eval_proto.get("enabled", False))
    fixed_res = eval_proto.get("fixed_resolution")
    train_res_param = eval_proto.get("train_resolution_param", "resolution")
    
    score_eval_attr = eval_proto.get("score_eval_attr", "score_eval_fixed")
    loss_eval_attr = eval_proto.get("loss_eval_attr", "loss_eval_fixed")

    rules = data.get("validation_rules", {})
    rules_enabled = bool(rules.get("enabled", False))
    score_min = rules.get("score_min")
    loss_min = rules.get("loss_min")
    max_epoch_jump = rules.get("max_epoch_jump")

    hpo_config = {
        "metric_loss_label": loss_label,
        "metric_score_label": score_label,
        "eval_protocol": {
            "enabled": eval_proto_enabled,
            "fixed_resolution": fixed_res,
            "train_resolution_param": train_res_param,
            "fixed_score_attr": score_eval_attr,
            "fixed_loss_attr": loss_eval_attr,
            "score_train_label": f"{score_label} (train)",
            "score_fixed_label": f"{score_label} (eval)",
            "use_fixed_metric_for_pruning": eval_proto_enabled,
            "prune_min_epoch": 5,
            "prune_compare_same_resolution_only": True,
            "prune_exclude_low_res_from_baseline": True,
            "pareto_deploy_resolution_only": True,
        },
        "validation_rules": {
            "enabled": rules_enabled,
            "score_min": float(score_min) if score_min is not None else None,
            "loss_min": float(loss_min) if loss_min is not None else None,
            "max_epoch_jump": float(max_epoch_jump) if max_epoch_jump is not None else None,
        },
        "param_labels": {},
        "manifest_metrics": metrics
    }
    
    return hpo_config
