"""Tests for the quickstart demo endpoint — the zero-friction onboarding wizard reached from the dashboard."""

import pytest
from unittest.mock import patch
import optuna
from src.db_manager import DATABASE_URL


DEMO_STUDY = "demo_segmentation_study"


@pytest.fixture(autouse=True)
def mock_run_training_worker():
    """Mock out the simulation worker to prevent spawning a background thread that pollutes/locks the DB."""
    with patch("simulators.training_worker.run_training_worker") as mock:
        yield mock


def test_quickstart_demo_creates_study(client):
    """First call to /api/quickstart_demo registers the study and returns success."""
    resp = client.post("/api/quickstart_demo")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("success") is True
    assert data["study_name"] == DEMO_STUDY

    study = optuna.load_study(study_name=DEMO_STUDY, storage=DATABASE_URL)
    assert study is not None


def test_quickstart_demo_idempotent(client):
    """Calling /api/quickstart_demo again returns success without error."""
    client.post("/api/quickstart_demo")
    resp = client.post("/api/quickstart_demo")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("success") is True
    assert data["study_name"] == DEMO_STUDY


def test_quickstart_demo_study_visible_in_config(client):
    """After the demo is initialized, the study details endpoint returns config."""
    client.post("/api/quickstart_demo")

    resp = client.get(f"/api/hpo_config?study_name={DEMO_STUDY}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "metric_loss_label" in data or "metric_score_label" in data


def test_cli_quickstart_wizard_creates_files(monkeypatch, tmp_path):
    """The interactive CLI quickstart wizard generates manifest and worker, and registers the study."""
    import os
    import optuna
    import yaml
    from unittest.mock import patch
    from hpo_cli import cmd_quickstart
    from src.db_manager import DATABASE_URL

    # Change working directory to a temporary path so we don't pollute the workspace
    monkeypatch.chdir(tmp_path)

    # Mock inputs:
    # 1. Study name base -> "cli_quickstart_test"
    # 2. Hyperparameter name -> "my_param"
    # 3. Min bound -> "-5.0"
    # 4. Max bound -> "5.0"
    # 5. Direction -> "maximize"
    inputs = iter(["cli_quickstart_test", "my_param", "-5.0", "5.0", "maximize"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    # Catch the sys.exit(0) call to prevent pytest from exiting
    with patch("sys.exit") as mock_exit:
        cmd_quickstart(args=None)

    # Verify sys.exit was called with 0
    mock_exit.assert_called_once_with(0)

    # Verify files were generated in the temp directory
    yaml_path = os.path.join(tmp_path, "quickstart.hpo.yaml")
    worker_path = os.path.join(tmp_path, "quickstart_worker.py")
    assert os.path.exists(yaml_path)
    assert os.path.exists(worker_path)

    # Verify YAML content structure
    with open(yaml_path) as f:
        manifest_data = yaml.safe_load(f)
    assert manifest_data["study_name"] == "cli_quickstart_test"
    assert manifest_data["metrics"]["primary_score"] == "score"
    assert manifest_data["params"][0]["name"] == "my_param"
    assert manifest_data["params"][0]["min"] == -5.0
    assert manifest_data["params"][0]["max"] == 5.0

    # Verify database study was registered
    study = optuna.load_study(study_name="cli_quickstart_test", storage=DATABASE_URL)
    assert study is not None

