"""Tests for the MCP server tools — verify unified error handling, return shapes, and edge cases."""

import yaml

from hpo_mcp_server import (
    get_study_data,
    get_study_cards,
    validate_manifest,
    init_from_manifest,
    export_manifest,
    resource_grill,
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
# get_study_data
# ---------------------------------------------------------------------------

def test_get_study_data_valid_study_with_completed_trial(client, initialized_study):
    """Return a valid packet with counts and health when study has completed trials."""
    study_name = initialized_study

    resp = client.post(
        "/api/suggest_trial",
        json={"study_name": study_name, "worker_id": "w1"},
    )
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]

    resp = client.post(
        "/api/complete_trial",
        json={
            "study_name": study_name,
            "trial_id": trial_id,
            "worker_id": "w1",
            "epoch": 1,
            "score": 0.80,
            "loss": 0.20,
            "weights_path": "model.pt",
            "history": [{"epoch": 1, "score": 0.80, "loss": 0.20}],
            "state": "COMPLETE",
        },
    )
    assert resp.status_code == 200

    result = get_study_data(study_name)
    assert isinstance(result, dict)
    assert result.get("study_name") == study_name
    assert "counts" in result
    assert result["counts"].get("complete", 0) >= 1
    assert "trial_bins" in result
    assert "health" in result


def test_get_study_data_nonexistent_study():
    """Returns success=False, error when study doesn't exist."""
    result = get_study_data("nonexistent_study_xyz")
    assert isinstance(result, dict)
    assert result.get("success") is False
    assert "error" in result


def test_get_study_data_empty_study(client, initialized_study):
    """Returns a valid packet with zero completed trials for a fresh study."""
    result = get_study_data(initialized_study)
    assert isinstance(result, dict)
    assert result.get("study_name") == initialized_study
    assert result.get("counts", {}).get("complete") == 0


# ---------------------------------------------------------------------------
# get_study_cards
# ---------------------------------------------------------------------------

def test_get_study_cards_valid_study_no_cards(initialized_study):
    """Returns success=True, cards=[] when study exists but has no cards."""
    result = get_study_cards(initialized_study)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert isinstance(result.get("cards"), list)
    assert result.get("cards") == []


def test_get_study_cards_nonexistent_study():
    """Returns success=False, error when study doesn't exist."""
    result = get_study_cards("nonexistent_study_xyz")
    assert isinstance(result, dict)
    assert result.get("success") is False
    assert "error" in result
    assert "not found" in result["error"]


def test_get_study_cards_no_argument():
    """Returns success=True with a list when no study_name is passed."""
    result = get_study_cards()
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert isinstance(result.get("cards"), list)


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
# export_manifest
# ---------------------------------------------------------------------------

def test_export_manifest_valid_study():
    """Returns success=True with parseable YAML string for an existing study."""
    data = yaml.safe_load(VALID_MANIFEST_YAML)
    study_name = "mcp_test_export_valid"
    data["study_name"] = study_name
    yaml_str = yaml.dump(data)
    init_from_manifest(yaml_str, force=True)

    result = export_manifest(study_name)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert "yaml_str" in result
    yaml_str = result["yaml_str"]
    assert isinstance(yaml_str, str)
    assert len(yaml_str) > 0

    reparsed = yaml.safe_load(yaml_str)
    assert reparsed.get("study_name") == study_name


def test_export_manifest_nonexistent_study():
    """Returns success=False with error for a non-existent study (no longer raises)."""
    result = export_manifest("nonexistent_study_xyz")
    assert isinstance(result, dict)
    assert result.get("success") is False
    assert "error" in result
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# resource_grill
# ---------------------------------------------------------------------------

def test_resource_grill_static_content():
    """Resource returns a non-empty string containing expected onboarding keywords."""
    content = resource_grill()
    assert isinstance(content, str)
    assert len(content) > 0
    assert "validate_manifest" in content
    assert "AGENTS.md" in content
