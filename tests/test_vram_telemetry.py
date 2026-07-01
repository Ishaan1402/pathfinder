import pytest
from optuna.trial import TrialState
from src.analytics import build_study_packet


def test_vram_telemetry_flows_to_study_packet(client, initialized_study):
    """
    Submits a trial with VRAM data and verifies the review packet contains
    populated VRAM telemetry rather than empty defaults.
    """
    study_name = initialized_study

    # Submit a trial with explicit GPU/VRAM data
    resp = client.post(
        "/api/suggest_trial",
        json={"study_name": study_name, "worker_id": "vram_test_worker"},
    )
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]

    payload = {
        "study_name": study_name,
        "trial_id": trial_id,
        "worker_id": "vram_test_worker",
        "epoch": 5,
        "score": 0.85,
        "loss": 0.14,
        "weights_path": "model.pt",
        "history": [{"epoch": 5, "score": 0.85, "loss": 0.14}],
        "state": "COMPLETE",
        "gpu_model": "NVIDIA A100",
        "max_vram_gb": 40.0,
        "oom_triggered": False,
    }
    resp = client.post("/api/complete_trial", json=payload)
    assert resp.status_code == 200

    # Fetch the review packet
    packet = build_study_packet(study_name)
    vram = packet.get("vram_telemetry", {})

    assert vram.get("gpu_model") == "NVIDIA A100", (
        f"Expected GPU model 'NVIDIA A100', got {vram.get('gpu_model')}"
    )
    assert vram.get("gpu_capacity_gb", 0) > 0, (
        f"Expected gpu_capacity_gb > 0, got {vram.get('gpu_capacity_gb')}"
    )
    assert vram.get("oom_count", -1) >= 0, (
        f"Expected oom_count >= 0, got {vram.get('oom_count')}"
    )
