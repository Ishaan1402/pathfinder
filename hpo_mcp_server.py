from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Pathfinder")

from src.db_manager import init_db

# --- MCP TOOLS ---

@mcp.tool()
def get_study_data(study_name: str) -> Dict[str, Any]:
    """Returns the compacted HPO review packet, utilizing a lazy materialization cache layer."""
    from src.analytics import build_study_packet
    return build_study_packet(study_name)


@mcp.tool()
def get_study_cards(study_name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves generated study cards (model cards, recaps) from the database to enable cross-study queries."""
    from src.analytics import load_study_cards

    if study_name is not None:
        import optuna
        from src.db_manager import DATABASE_URL
        try:
            optuna.load_study(study_name=study_name, storage=DATABASE_URL)
        except KeyError:
            return {"success": False, "error": f"Study '{study_name}' not found."}

    cards = load_study_cards(study_name)
    return {"success": True, "cards": cards}


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
    from src.onboarding import init_study_from_manifest_dict

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
def export_manifest(study_name: str) -> Dict[str, Any]:
    """Export the active search space, HPO config, and context of an existing study as a valid manifest YAML string."""
    from src.manifest import export_manifest_yaml
    try:
        result = export_manifest_yaml(study_name)
        return {"success": True, "yaml_str": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- MCP PROMPT RESOURCES ---

@mcp.resource("hpo://prompts/grill")
def resource_grill() -> str:
    """Onboarding checklist: interview, then manifest loop."""
    return """# Pathfinder Onboarding (Grill + Manifest Loop)

See AGENTS.md for the full procedure. After interviewing the user (metrics, GPU, bounds, hypothesis):

1. Draft a YAML manifest configuration (e.g. `train.hpo.yaml`).
2. Call `validate_manifest(yaml_str)` to check for errors/warnings mechanically.
3. Call `init_from_manifest(yaml_str)` to register the study in SQLite and Optuna.
4. Call `get_study_data(study_name)` to confirm the study is accessible and healthy.

Worker integration reference: `docs/INTEGRATION.md`. Do not write json space config files to disk.
"""


if __name__ == "__main__":
    init_db()
    mcp.run()
