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
        "fixed_score_attr": "score_eval_fixed",
        "fixed_loss_attr": "loss_eval_fixed",
        "score_train_label": "Score (train)",
        "score_fixed_label": "Score (eval)",
        "use_fixed_metric_for_pruning": True,
        "prune_min_epoch": 5,
        "prune_compare_same_resolution_only": True,
        "prune_exclude_low_res_from_baseline": True,
        "pareto_deploy_resolution_only": True,
    },
    "param_labels": {},
}


def load_hpo_config(study_name: Optional[str] = None) -> Dict[str, Any]:
    from .settings import settings
    if not study_name:
        study_name = settings.study_name
    if not study_name or not study_name.strip():
        return copy.deepcopy(DEFAULT_HPO_CONFIG)

    # Try loading study-specific config from DB
    data = None
    try:
        from .db_manager import get_db_session
        from .schema import SystemConfiguration
        with get_db_session() as session:
            row = session.query(SystemConfiguration).filter_by(
                study_name=study_name, config_key="hpo_config"
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
            save_hpo_config(data, study_name)
        except Exception as e:
            logger.warning(f"Failed to seed hpo_config in DB: {e}")

    defaults = DEFAULT_HPO_CONFIG

    try:
        merged = copy.deepcopy(defaults)
        merged.update({k: v for k, v in data.items() if k not in ("eval_protocol", "param_labels", "validation_rules")})
        merged["eval_protocol"] = {**defaults.get("eval_protocol", {}), **data.get("eval_protocol", {})}
        merged["validation_rules"] = {**defaults.get("validation_rules", {}), **data.get("validation_rules", {})}
        merged["param_labels"] = {**defaults.get("param_labels", {}), **data.get("param_labels", {})}
        return merged
    except Exception:
        return copy.deepcopy(defaults)


def save_hpo_config(config: Dict[str, Any], study_name: Optional[str] = None) -> None:
    from .settings import settings
    if not study_name:
        study_name = settings.study_name
    if not study_name or not study_name.strip():
        raise ValueError("study_name cannot be empty.")

    config = dict(config)

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

    # Bust cached study packets — config changes invalidate analytics
    try:
        from .db_manager import get_db_session
        from .schema import CompactedPacket
        with get_db_session() as session:
            session.query(CompactedPacket).filter_by(study_name=study_name).delete()
    except Exception:
        pass


def param_display_name(param: str, config: Optional[Dict[str, Any]] = None, study_name: Optional[str] = None) -> str:
    config = config or load_hpo_config(study_name)
    return config.get("param_labels", {}).get(param, param)


def normalize_trial_params(params: Dict[str, Any], config: Optional[Dict[str, Any]] = None, study_name: Optional[str] = None) -> Dict[str, Any]:
    return dict(params)

