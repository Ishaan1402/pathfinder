"""App-level HPO configuration (eval protocol, display labels) — separate from search space bounds.

Configuration is persisted in SQLite (system_configuration); there is no on-disk config file.
"""
import os
import json
from typing import Any, Dict, Optional

DEFAULT_HPO_CONFIG: Dict[str, Any] = {
    "metric_loss_label": "BCE",
    "metric_score_label": "Dice",
    "desktop_notifications_enabled": False,
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
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")

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
                data = json.loads(row.config_value)
    except Exception as e:
        print(f"Error loading hpo_config from DB: {e}")

    # Fallback to loading default template from disk if DB fails or has no config
    if not data:
        data = DEFAULT_HPO_CONFIG
        # Seed it into DB for this study so it exists in DB
        try:
            save_hpo_config(DEFAULT_HPO_CONFIG, study_name)
        except Exception:
            pass

    try:
        merged = json.loads(json.dumps(DEFAULT_HPO_CONFIG))
        merged.update({k: v for k, v in data.items() if k != "eval_protocol"})
        merged["eval_protocol"] = {**DEFAULT_HPO_CONFIG["eval_protocol"], **data.get("eval_protocol", {})}
        merged["param_labels"] = {**DEFAULT_HPO_CONFIG.get("param_labels", {}), **data.get("param_labels", {})}
        merged["legacy_param_aliases"] = {
            **DEFAULT_HPO_CONFIG.get("legacy_param_aliases", {}),
            **data.get("legacy_param_aliases", {}),
        }
        merged["legacy_capacity_values"] = {
            **DEFAULT_HPO_CONFIG.get("legacy_capacity_values", {}),
            **data.get("legacy_capacity_values", {}),
        }
        return merged
    except Exception:
        return json.loads(json.dumps(DEFAULT_HPO_CONFIG))


def save_hpo_config(config: Dict[str, Any], study_name: Optional[str] = None) -> None:
    if not study_name:
        study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")

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

