"""End-to-end integration test: broker startup, worker lifecycle, study packet, auth.

The broker is started in a background thread for the duration of this test only.
Uses the same SQLite database that conftest.py configures (a temp file).
"""
import os
import sys
import json
import threading
import time
import requests
import uvicorn


BROKER_PORT = 8123
BROKER_URL = f"http://127.0.0.1:{BROKER_PORT}"

MANIFEST_BASE = {
    "study_name": "unet_crack_segmentation_test",
    "metrics": {
        "primary_score": "score",
        "objectives": [
            {"name": "loss", "direction": "minimize", "label": "Loss"},
            {"name": "score", "direction": "maximize", "label": "Score"},
        ],
    },
    "params": [
        {"name": "learning_rate", "type": "float_log", "min": 1e-5, "max": 1e-2},
        {"name": "batch_size", "type": "categorical", "options": [2, 4, 8, 16]},
        {"name": "resolution", "type": "categorical", "options": [256, 512, 1024]},
        {"name": "model_capacity", "type": "categorical", "options": ["narrow", "wide"]},
        {"name": "loss_weight_ratio", "type": "float", "min": 0.0, "max": 1.0},
    ],
}


def _start_broker():
    from broker import app
    from src.db_manager import init_db
    init_db()
    uvicorn.run(app, host="127.0.0.1", port=BROKER_PORT, log_level="warning")


def test_integration():
    os.environ["HPO_BROKER_URL"] = BROKER_URL

    broker_thread = threading.Thread(target=_start_broker, daemon=True)
    broker_thread.start()
    time.sleep(2.0)

    study_name = MANIFEST_BASE["study_name"]

    # ---- Step 1: Init study from manifest ----
    from src.onboarding import init_study_from_manifest_dict

    manifest = json.loads(json.dumps(MANIFEST_BASE))
    init_study_from_manifest_dict(manifest, force=True)

    from src.db_manager import get_db_session
    from src.schema import SystemConfiguration, StudyStatus

    with get_db_session() as session:
        row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        assert row is not None, "Search space should be persisted"
        status = session.query(StudyStatus).filter_by(study_name=study_name).first()
        assert status is not None, "StudyStatus should exist"

    # ---- Step 2: Run a single trial via HTTP ----
    from src.hpo_client import TrialSession

    sess = TrialSession(broker_url=BROKER_URL, study_name=study_name)
    trial = sess.suggest()
    assert "params" in trial
    sess.complete(epoch=0, score=0.5, loss=0.5)

    # ---- Step 3: Simulator worker (5 trials) ----
    from simulators.training_worker import run_training_worker

    run_training_worker(
        study_name=study_name,
        max_trials=5,
        epochs_per_trial=5,
        broker_url=BROKER_URL,
    )

    # ---- Step 4: Study packet ----
    from src.analytics import build_study_packet

    packet = build_study_packet(study_name)
    assert "trial_bins" in packet
    assert "fanova_importances" in packet
    assert "vram_telemetry" in packet
    assert "health" in packet

    # ---- Step 5: Authentication ----
    os.environ["HPO_SECRET_TOKEN"] = "test_integration_token"

    resp = requests.get(f"{BROKER_URL}/api/study_details?study_name={study_name}")
    assert resp.status_code == 401

    resp = requests.get(
        f"{BROKER_URL}/api/study_details?study_name={study_name}",
        headers={"X-HPO-Token": "test_integration_token"},
    )
    assert resp.status_code == 200

    resp = requests.get(
        f"{BROKER_URL}/api/study_details?study_name={study_name}",
        headers={"Authorization": "Bearer test_integration_token"},
    )
    assert resp.status_code == 200

    del os.environ["HPO_SECRET_TOKEN"]

    # ---- Step 6: CLI status ----
    import subprocess
    result = subprocess.run(
        [sys.executable, "hpo_cli.py", "status", "--study", study_name],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "STUDY STATUS" in result.stdout

    # ---- Step 7: Study cards endpoint ----
    resp = requests.get(f"{BROKER_URL}/api/study_cards?study_name={study_name}")
    data = resp.json()
    assert data["success"]
