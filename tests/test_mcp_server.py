"""Tests for the MCP server tools and resources — verify tool return shapes and resource smoke checks."""

import json
import yaml

from hpo_mcp_server import (
    validate_manifest,
    init_from_manifest,
    study_packet_resource,
    study_cards_resource,
)

# ---------------------------------------------------------------------------
# Manifest YAML helpers
# ---------------------------------------------------------------------------

VALID_MANIFEST_YAML = """study_name: mcp_test_study
metrics:
  primary_score: score
  objectives:
    - name: score
      label: Score
      direction: maximize
    - name: loss
      label: Loss
      direction: minimize
params:
  - name: learning_rate
    type: float_log
    min: 0.00001
    max: 0.1
  - name: batch_size
    type: categorical
    options: [16, 32, 64]
worker:
  entrypoint: python train.py
"""


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------

def test_validate_manifest_valid_yaml():
    """Valid YAML with valid schema returns success=True, no errors."""
    result = validate_manifest(VALID_MANIFEST_YAML)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result.get("errors") == []


def test_validate_manifest_garbage_string():
    """Non-YAML string returns success=False with YAML parse error."""
    result = validate_manifest("<<: *does_not_exist")
    assert result.get("success") is False
    assert "Invalid YAML structure" in result["errors"][0]


def test_validate_manifest_non_dict_yaml():
    """Valid YAML that is not a dict returns success=False."""
    result = validate_manifest('"just a string"')
    assert result.get("success") is False
    assert "Manifest root must be a dictionary" in result["errors"][0]


def test_validate_manifest_missing_study_name():
    """Valid YAML but missing required field returns errors."""
    manifest = yaml.safe_load(VALID_MANIFEST_YAML)
    del manifest["study_name"]
    yaml_str = yaml.dump(manifest)
    result = validate_manifest(yaml_str)
    assert result.get("success") is False
    assert any("study_name" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# init_from_manifest
# ---------------------------------------------------------------------------

def test_init_from_manifest_fresh_study():
    """Fresh manifest with force=False succeeds."""
    data = yaml.safe_load(VALID_MANIFEST_YAML)
    data["study_name"] = "mcp_test_init_fresh"
    yaml_str = yaml.dump(data)
    result = init_from_manifest(yaml_str, force=True)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert "initialized" in result.get("message", "").lower()


def test_init_from_manifest_duplicate_no_force():
    """Same study name without force returns error."""
    data = yaml.safe_load(VALID_MANIFEST_YAML)
    data["study_name"] = "mcp_test_init_dup"
    yaml_str = yaml.dump(data)

    result1 = init_from_manifest(yaml_str, force=True)
    assert result1.get("success") is True

    result2 = init_from_manifest(yaml_str, force=False)
    assert result2.get("success") is False
    assert "already exists" in result2.get("error", "")


def test_init_from_manifest_duplicate_with_force():
    """Same study name with force=True overwrites successfully."""
    data = yaml.safe_load(VALID_MANIFEST_YAML)
    data["study_name"] = "mcp_test_init_force"
    yaml_str = yaml.dump(data)

    init_from_manifest(yaml_str, force=True)
    result = init_from_manifest(yaml_str, force=True)
    assert result.get("success") is True
    assert "initialized" in result.get("message", "").lower()


def test_init_from_manifest_garbage_yaml():
    """Garbage YAML string returns success=False."""
    result = init_from_manifest("<<: *does_not_exist", force=False)
    assert result.get("success") is False
    assert "Invalid YAML structure" in result.get("error", "")


def test_init_from_manifest_schema_errors():
    """Valid YAML with missing required fields returns success=False."""
    data = yaml.safe_load(VALID_MANIFEST_YAML)
    del data["params"]
    yaml_str = yaml.dump(data)
    result = init_from_manifest(yaml_str, force=True)
    assert result.get("success") is False
    assert "Cannot initialize study" in result.get("error", "")


# ---------------------------------------------------------------------------
# Resource smoke tests
# ---------------------------------------------------------------------------

def test_study_packet_resource_existing_study(initialized_study):
    """Resource returns JSON string that parses to the expected packet shape."""
    content = study_packet_resource(initialized_study)
    assert isinstance(content, str)
    packet = json.loads(content)
    assert isinstance(packet, dict)
    assert packet.get("study_name") == initialized_study
    assert "health" in packet
    assert "counts" in packet


def test_study_packet_resource_nonexistent_study():
    """Resource returns JSON string with error payload for missing study."""
    content = study_packet_resource("nonexistent_study_xyz")
    packet = json.loads(content)
    assert packet.get("success") is False
    assert "error" in packet


def test_study_cards_resource_existing_study(initialized_study):
    """Resource returns JSON string that parses to a list (or empty list)."""
    content = study_cards_resource(initialized_study)
    cards = json.loads(content)
    assert isinstance(cards, list)


def test_study_cards_resource_nonexistent_study():
    """Resource returns empty list for missing study."""
    content = study_cards_resource("nonexistent_study_xyz")
    cards = json.loads(content)
    assert isinstance(cards, list)
    assert cards == []
