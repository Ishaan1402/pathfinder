"""App-level HPO configuration (eval protocol, display labels) — separate from search space bounds.

Configuration is persisted in SQLite (system_configuration); there is no on-disk config file.
"""

import json
import logging
import copy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Generic defaults for hyperparameter optimization configs.
# These defaults can be overridden per-study in the system_configuration database table.
DEFAULT_HPO_CONFIG: Dict[str, Any] = {
    "config_version": 2,
    "metric_loss_label": "Loss",
    "metric_score_label": "Score",
    "metric_names": {"score": "score", "loss": "loss"},
    "desktop_notifications_enabled": False,
    "validation_rules": {
        "score_min": None,
        "loss_min": None,
        "max_epoch_jump": None,
        "enabled": False,
    },
    "eval_protocol": {
        "enabled": False,
        "fixed_resolution": None,
        "train_resolution_param": "resolution",
        "fixed_dice_attr": "score_eval_fixed",
        "fixed_bce_attr": "loss_eval_fixed",
        "dice_train_label": "Score (train)",
        "dice_fixed_label": "Score (eval)",
        "use_fixed_metric_for_pruning": True,
        "prune_min_epoch": 5,
        "prune_compare_same_resolution_only": True,
        "prune_exclude_low_res_from_baseline": True,
        "pareto_deploy_resolution_only": True,
    },
    "param_labels": {},
}

# Legacy defaults for U-Net crack segmentation studies (config_version = 1).
LEGACY_DEFAULT_HPO_CONFIG: Dict[str, Any] = {
    "metric_loss_label": "BCE",
    "metric_score_label": "Dice",
    "desktop_notifications_enabled": False,
    "validation_rules": {
        "score_min": 0.0,
        "loss_min": 0.0,
        "max_epoch_jump": 0.5,
        "enabled": True,
    },
    "eval_protocol": {
        "enabled": False,
        "fixed_resolution": None,
        "train_resolution_param": "resolution",
        "fixed_dice_attr": "dice_eval_fixed",
        "fixed_bce_attr": "bce_eval_fixed",
        "dice_train_label": "Dice (train)",
        "dice_fixed_label": "Dice (eval)",
        "use_fixed_metric_for_pruning": True,
        "prune_min_epoch": 5,
        "prune_compare_same_resolution_only": True,
        "prune_exclude_low_res_from_baseline": True,
        "pareto_deploy_resolution_only": True,
        "low_train_res_warning": 384,
        "patch_size_below_512": 256,
        "patch_size_at_512_plus": 448,
    },
    "param_labels": {},
    "legacy_param_aliases": {"encoder_name": "model_capacity"},
    "legacy_capacity_values": {
        "resnet34": "narrow",
        "efficientnet-b0": "narrow",
        "resnet50": "wide",
    },
}


def load_hpo_config(study_name: Optional[str] = None) -> Dict[str, Any]:
    from .settings import settings
    if not study_name:
        study_name = settings.study_name

    # Try loading study-specific config from DB
    data = None
    try:
        from .db_manager import get_db_session
        from .schema import SystemConfiguration
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="hpo_config"
            ).first()
            if not row and study_name != "_global":
                # Fallback to _global config in DB
                row = session.query(SystemConfiguration).filter_by(
                    study_name="_global", config_key="hpo_config"
                ).first()
            if row:
                loaded_data = json.loads(row.config_value)
                # If we fell back to _global and the current study is NOT a legacy U-Net study,
                # check if the global config is legacy (version 1). If it is, ignore it to
                # prevent legacy poisoning of generic studies.
                if row.study_name == "_global" and study_name not in ("seg_v1", "bridge_crack_study"):
                    if loaded_data.get("config_version", 1) == 1:
                        loaded_data = None
                if loaded_data:
                    data = loaded_data
    except Exception as e:
        print(f"Error loading hpo_config from DB: {e}")

    # Fallback to loading default template from disk if DB fails or has no config
    if not data:
        is_legacy = study_name in ("seg_v1", "bridge_crack_study")
        data = LEGACY_DEFAULT_HPO_CONFIG if is_legacy else DEFAULT_HPO_CONFIG
        # Seed it into DB for this study so it exists in DB
        try:
            save_hpo_config(data, study_name)
        except Exception as e:
            logger.warning(f"Failed to seed hpo_config in DB: {e}")

    is_legacy_name = study_name in ("seg_v1", "bridge_crack_study")
    config_version = data.get("config_version", 1 if is_legacy_name else 2)
    defaults = LEGACY_DEFAULT_HPO_CONFIG if config_version == 1 else DEFAULT_HPO_CONFIG

    try:
        merged = copy.deepcopy(defaults)
        merged.update({k: v for k, v in data.items() if k not in ("eval_protocol", "param_labels", "legacy_param_aliases", "legacy_capacity_values", "validation_rules")})
        merged["eval_protocol"] = {**defaults.get("eval_protocol", {}), **data.get("eval_protocol", {})}
        merged["validation_rules"] = {**defaults.get("validation_rules", {}), **data.get("validation_rules", {})}
        merged["param_labels"] = {**defaults.get("param_labels", {}), **data.get("param_labels", {})}
        if "legacy_param_aliases" in defaults or "legacy_param_aliases" in data:
            merged["legacy_param_aliases"] = {
                **defaults.get("legacy_param_aliases", {}),
                **data.get("legacy_param_aliases", {}),
            }
        if "legacy_capacity_values" in defaults or "legacy_capacity_values" in data:
            merged["legacy_capacity_values"] = {
                **defaults.get("legacy_capacity_values", {}),
                **data.get("legacy_capacity_values", {}),
            }
        return merged
    except Exception:
        return copy.deepcopy(defaults)


def save_hpo_config(config: Dict[str, Any], study_name: Optional[str] = None) -> None:
    from .settings import settings
    if not study_name:
        study_name = settings.study_name

    # Enforce config_version on save to prevent client downgrades
    config = dict(config)
    is_legacy = study_name in ("seg_v1", "bridge_crack_study")
    config["config_version"] = config.get("config_version", 1 if is_legacy else 2)

    try:
        from .db_manager import get_db_session
        from .schema import SystemConfiguration
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="hpo_config"
            ).first()
            if row:
                row.config_value = json.dumps(config)
                row.version += 1
            else:
                session.add(SystemConfiguration(
                    study_name=study_name,
                    config_key="hpo_config",
                    config_value=json.dumps(config),
                    version=1
                ))
    except Exception as e:
        print(f"Error saving hpo_config to DB: {e}")


def param_display_name(param: str, config: Optional[Dict[str, Any]] = None, study_name: Optional[str] = None) -> str:
    config = config or load_hpo_config(study_name)
    return config.get("param_labels", {}).get(param, param)


def normalize_trial_params(params: Dict[str, Any], config: Optional[Dict[str, Any]] = None, study_name: Optional[str] = None) -> Dict[str, Any]:
    """Map legacy param names/values for display and workers."""
    config = config or load_hpo_config(study_name)
    out = dict(params)
    for old, new in config.get("legacy_param_aliases", {}).items():
        if old in out and new not in out:
            val = out.pop(old)
            mapped = config.get("legacy_capacity_values", {}).get(val, val)
            out[new] = mapped
    return out

