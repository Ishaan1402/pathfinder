"""Tests for the quickstart demo endpoint — the zero-friction onboarding wizard reached from the dashboard."""

import optuna
from src.db_manager import DATABASE_URL


DEMO_STUDY = "demo_segmentation_study"


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
