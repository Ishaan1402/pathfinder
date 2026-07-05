"""Auth and lease-ownership tests for the broker HTTP surface."""
import uuid


def test_report_requires_lease_ownership(client, initialized_study):
    owner = str(uuid.uuid4())
    intruder = str(uuid.uuid4())

    sug = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": owner})
    assert sug.status_code == 200, sug.text
    trial_id = sug.json()["trial_id"]

    # Intruder (no lease) cannot report.
    bad = client.post(
        "/api/report_epoch",
        json={"study_name": initialized_study, "trial_id": trial_id, "worker_id": intruder,
              "epoch": 0, "score": 0.5, "loss": 0.5},
    )
    assert bad.status_code == 403

    # Missing worker_id also cannot report.
    nobody = client.post(
        "/api/report_epoch",
        json={"study_name": initialized_study, "trial_id": trial_id,
              "epoch": 0, "score": 0.5, "loss": 0.5},
    )
    assert nobody.status_code == 403

    # Owner can report.
    ok = client.post(
        "/api/report_epoch",
        json={"study_name": initialized_study, "trial_id": trial_id, "worker_id": owner,
              "epoch": 0, "score": 0.5, "loss": 0.5},
    )
    assert ok.status_code == 200, ok.text


def test_complete_requires_lease_for_inflight_trial(client, initialized_study):
    owner = str(uuid.uuid4())
    intruder = str(uuid.uuid4())

    sug = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": owner})
    trial_id = sug.json()["trial_id"]

    bad = client.post(
        "/api/complete_trial",
        json={"study_name": initialized_study, "trial_id": trial_id, "worker_id": intruder,
              "epoch": 1, "score": 0.6, "loss": 0.4, "weights_path": "m.pt",
              "history": [{"epoch": 1, "score": 0.6, "loss": 0.4}], "state": "COMPLETE"},
    )
    assert bad.status_code == 403
