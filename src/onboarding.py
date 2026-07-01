import json
import logging
import optuna
from typing import Dict, Any, List, Optional

from src.db_manager import get_db_session, DATABASE_URL

logger = logging.getLogger(__name__)
from src.schema import SystemConfiguration, StudyStatus

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
    try:
        # Create Optuna study
        if multi_objective:
            study_directions = directions or ["minimize", "maximize"]
            optuna.create_study(
                study_name=study_name,
                storage=DATABASE_URL,
                directions=study_directions,
                load_if_exists=False
            )
        else:
            study_direction = directions[0] if directions else "maximize"
            optuna.create_study(
                study_name=study_name,
                storage=DATABASE_URL,
                direction=study_direction,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
                load_if_exists=False
            )

        # Persist configurations to database
        with get_db_session() as session:
            # Active Search Space
            session.merge(SystemConfiguration(
                study_name=study_name,
                config_key="active_search_space",
                config_value=json.dumps(active_search_space)
            ))
            # HPO Config (global key fallback or study-specific)
            session.merge(SystemConfiguration(
                study_name=study_name,
                config_key="hpo_config",
                config_value=json.dumps(hpo_config)
            ))
            # Project Context / Hypothesis
            session.merge(SystemConfiguration(
                study_name=study_name,
                config_key="project_context",
                config_value=json.dumps(project_context or {})
            ))
            # Source Files Audit Trail
            if source_files:
                session.merge(SystemConfiguration(
                    study_name=study_name,
                    config_key="source_files",
                    config_value=json.dumps(source_files)
                ))
            
            # Initial healthy status
            session.merge(StudyStatus(
                study_name=study_name,
                health_tier="healthy",
                health_reason="Study initialized successfully."
            ))

        return f"Study '{study_name}' successfully initialized and configured in database."
    except Exception as e:
        raise RuntimeError(f"Failed to initialize study '{study_name}': {str(e)}") from e


def delete_study_internal(study_name: str, confirm: bool = False) -> Dict[str, Any]:
    """Permanently delete a study: its Optuna trials and ALL custom metadata rows."""
    if not confirm:
        return {"success": False, "error": "Refusing to delete without confirm=True."}

    deleted: Dict[str, Any] = {"optuna_study": False, "rows": {}}

    # 1. Optuna's own study (trials, params, distributions).
    try:
        optuna.delete_study(study_name=study_name, storage=DATABASE_URL)
        deleted["optuna_study"] = True
    except KeyError:
        pass  # already absent
    except Exception as e:
        return {"success": False, "error": f"Failed to delete Optuna study: {e}"}

    # 2. Every custom table that carries a study_name column.
    from src.schema import Base
    from sqlalchemy import delete as sa_delete
    try:
        with get_db_session() as session:
            for table in Base.metadata.sorted_tables:
                if "study_name" in table.c:
                    result = session.execute(
                        sa_delete(table).where(table.c.study_name == study_name)
                    )
                    if result.rowcount:
                        deleted["rows"][table.name] = result.rowcount
    except Exception as e:
        return {"success": False, "error": f"Failed to delete metadata rows: {e}"}

    return {"success": True, "study_name": study_name, "deleted": deleted}


def init_study_from_manifest_dict(data: Dict[str, Any], force: bool = False) -> str:
    """Validate manifest and register a study, cleaning up completely on force=True."""
    from src.manifest import validate_manifest, _manifest_params_to_search_space, _manifest_to_hpo_config
    
    errors, warnings = validate_manifest(data)
    if errors:
        raise ValueError("Cannot initialize study — manifest has errors: " + "; ".join(errors))

    study_name = data["study_name"]

    # Check if study name already exists in DB or Optuna
    study_exists = False
    try:
        optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        study_exists = True
    except KeyError:
        try:
            with get_db_session() as session:
                row = session.query(SystemConfiguration).filter_by(
                    study_name=study_name, config_key="active_search_space"
                ).first()
                if row:
                    study_exists = True
        except Exception as e:
            logger.warning(f"Failed to check study existence in DB: {e}")
    except Exception as e:
        logger.warning(f"Failed to check Optuna study existence: {e}")

    if study_exists and not force:
        raise ValueError(f"Study '{study_name}' already exists. Refusing to initialize. Use --force to overwrite.")

    if study_exists and force:
        # Call the thorough delete_study tool to purge all trials and metadata
        result = delete_study_internal(study_name=study_name, confirm=True)
        if not result.get("success"):
            raise RuntimeError(
                f"Failed to delete existing study '{study_name}': {result.get('error', 'unknown error')}"
            )

    metrics = data["metrics"]
    active_search_space = _manifest_params_to_search_space(data["params"])
    hpo_config = _manifest_to_hpo_config(data)
    project_context = data.get("project_context", {})
    
    # Track worker entrypoint/env in project context
    worker_data = data.get("worker", {})
    if worker_data.get("entrypoint"):
        project_context["worker_entrypoint"] = worker_data["entrypoint"]
    if worker_data.get("env"):
        project_context["worker_env"] = worker_data["env"]

    # Register the study
    result = initialize_study(
        study_name=study_name,
        active_search_space=active_search_space,
        hpo_config=hpo_config,
        project_context=project_context,
        source_files=data.get("source_files"),
        multi_objective=(len(metrics["objectives"]) > 1),
        directions=[obj["direction"] for obj in metrics["objectives"]]
    )
    return result
