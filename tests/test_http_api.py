"""HTTP-level tests for the broker worker contract (suggest -> report -> complete).

These pin the invariants that must hold across the security/refactor work: the worker
lifecycle round-trips, unknown trials 404, and completion is idempotent. Auth- and
lease-specific behavior is covered in test_http_auth.py.
"""
import uuid


def _suggest(client, study_name, worker_id):
    resp = client.post("/api/suggest_trial", json={"study_name": study_name, "worker_id": worker_id})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["success"] is True
    assert "trial_id" in data and "params" in data
    return data


def test_worker_lifecycle_records_result(client, initialized_study):
    worker_id = str(uuid.uuid4())
    sug = _suggest(client, initialized_study, worker_id)
    trial_id = sug["trial_id"]

    for epoch in range(3):
        r = client.post(
            "/api/report_epoch",
            json={
                "study_name": initialized_study,
                "trial_id": trial_id,
                "worker_id": worker_id,
                "epoch": epoch,
                "score": 0.5 + epoch * 0.1,
                "loss": 0.5 - epoch * 0.1,
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["should_prune"] in (True, False)

    c = client.post(
        "/api/complete_trial",
        json={
            "study_name": initialized_study,
            "trial_id": trial_id,
            "worker_id": worker_id,
            "epoch": 2,
            "score": 0.7,
            "loss": 0.3,
            "weights_path": "model.pt",
            "history": [{"epoch": 2, "dice": 0.7, "bce": 0.3}],
            "state": "COMPLETE",
        },
    )
    assert c.status_code == 200, c.text
    assert c.json()["success"] is True

    # A TrialResult row should now exist for this study.
    from src.db_manager import get_db_session
    from src.schema import TrialResult
    with get_db_session() as session:
        rows = session.query(TrialResult).filter_by(study_name=initialized_study).all()
        assert any(row.trial_id == trial_id for row in rows)


def test_delete_study_removes_optuna_and_metadata(client, initialized_study):
    from hpo_mcp_server import delete_study

    # Produce a trial + a TrialResult row.
    worker_id = str(uuid.uuid4())
    sug = _suggest(client, initialized_study, worker_id)
    client.post("/api/complete_trial", json={
        "study_name": initialized_study, "trial_id": sug["trial_id"], "worker_id": worker_id,
        "epoch": 1, "score": 0.6, "loss": 0.4, "weights_path": "m.pt",
        "history": [{"epoch": 1, "dice": 0.6, "bce": 0.4}], "state": "COMPLETE",
    })

    res = delete_study(initialized_study, confirm=False)
    assert res["success"] is False  # guarded

    res = delete_study(initialized_study, confirm=True)
    assert res["success"] is True
    assert res["deleted"]["optuna_study"] is True

    # Study is gone: the worker path now 404s, and no TrialResult rows remain.
    assert client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "w"}).status_code == 404
    from src.db_manager import get_db_session
    from src.schema import TrialResult
    with get_db_session() as session:
        assert session.query(TrialResult).filter_by(study_name=initialized_study).count() == 0


def test_suggest_on_uninitialized_study_returns_404(client):
    r = client.post(
        "/api/suggest_trial",
        json={"study_name": "totally_unknown_study_xyz", "worker_id": "w"},
    )
    assert r.status_code == 404


def test_report_unknown_trial_returns_404(client, initialized_study):
    r = client.post(
        "/api/report_epoch",
        json={
            "study_name": initialized_study,
            "trial_id": 999999,
            "worker_id": "nobody",
            "epoch": 0,
            "score": 0.1,
            "loss": 0.9,
        },
    )
    assert r.status_code == 404


def test_complete_is_idempotent(client, initialized_study):
    worker_id = str(uuid.uuid4())
    sug = _suggest(client, initialized_study, worker_id)
    trial_id = sug["trial_id"]

    payload = {
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": worker_id,
        "epoch": 1,
        "score": 0.6,
        "loss": 0.4,
        "weights_path": "model.pt",
        "history": [{"epoch": 1, "dice": 0.6, "bce": 0.4}],
        "state": "COMPLETE",
    }
    first = client.post("/api/complete_trial", json=payload)
    assert first.status_code == 200, first.text
    second = client.post("/api/complete_trial", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["success"] is True
