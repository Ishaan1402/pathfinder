import json
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

from src.db_manager import init_db

mcp = FastMCP("Pathfinder")


# --- MCP TOOLS (state-changing operations) ---

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


# --- MCP RESOURCES (data retrieval) ---

@mcp.resource("hpo://studies/{study_name}/packet")
def study_packet_resource(study_name: str) -> str:
    """Compacted HPO review packet: trial telemetry, health tier, fANOVA importances, OOM patterns."""
    from src.analytics import build_study_packet
    return json.dumps(build_study_packet(study_name), indent=2)


@mcp.resource("hpo://studies/{study_name}/cards")
def study_cards_resource(study_name: str) -> str:
    """Generated study cards (model cards, recaps) from the database."""
    from src.analytics import load_study_cards
    return json.dumps(load_study_cards(study_name), indent=2)


if __name__ == "__main__":
    init_db()
    mcp.run()
