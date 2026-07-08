"""Auth and lease-ownership tests for the broker HTTP surface."""

import os
import uuid

import pytest


# ---------------------------------------------------------------------------
# Lease ownership (worker-level auth)
# ---------------------------------------------------------------------------

def test_report_requires_lease_ownership(client, initialized_study):
    owner = str(uuid.uuid4())
    intruder = str(uuid.uuid4())

    sug = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": owner})
    assert sug.status_code == 200, sug.text
    trial_id = sug.json()["trial_id"]

    bad = client.post(
        "/api/report_epoch",
        json={"study_name": initialized_study, "trial_id": trial_id, "worker_id": intruder,
              "epoch": 0, "score": 0.5, "loss": 0.5},
    )
    assert bad.status_code == 403

    nobody = client.post(
        "/api/report_epoch",
        json={"study_name": initialized_study, "trial_id": trial_id,
              "epoch": 0, "score": 0.5, "loss": 0.5},
    )
    assert nobody.status_code == 403

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


# ---------------------------------------------------------------------------
# Token auth (broker middleware)
# ---------------------------------------------------------------------------

TEST_TOKEN = "test_auth_token_123"


@pytest.fixture
def enable_auth():
    os.environ["HPO_SECRET_TOKEN"] = TEST_TOKEN
    yield
    os.environ.pop("HPO_SECRET_TOKEN", None)


def test_bypass_routes_work_without_token(enable_auth, client):
    """Health, root, styles.css, and /api/login are explicitly bypassed."""
    assert client.get("/health").status_code == 200
    assert client.get("/styles.css").status_code != 401
    # POST /api/login requires a body; GET returns 405 but importantly NOT 401
    assert client.get("/api/login").status_code != 401


def test_protected_route_returns_401_without_token(enable_auth, client):
    """Protected API routes require a token."""
    resp = client.get("/api/hpo_config?study_name=does_not_exist")
    assert resp.status_code == 401, resp.text
    assert "Unauthorized" in resp.json()["error"]


def test_x_hpo_token_header_passes_auth(enable_auth, client, initialized_study):
    """X-HPO-Token header is accepted."""
    resp = client.get(
        f"/api/study_details?study_name={initialized_study}",
        headers={"X-HPO-Token": TEST_TOKEN},
    )
    assert resp.status_code == 200, resp.text


def test_authorization_bearer_header_passes_auth(enable_auth, client, initialized_study):
    """Authorization: Bearer <token> header is accepted."""
    resp = client.get(
        f"/api/study_details?study_name={initialized_study}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert resp.status_code == 200, resp.text


def test_authorization_header_without_bearer_prefix_passes_auth(enable_auth, client, initialized_study):
    """Authorization header without Bearer prefix falls through to raw comparison."""
    resp = client.get(
        f"/api/study_details?study_name={initialized_study}",
        headers={"Authorization": TEST_TOKEN},
    )
    assert resp.status_code == 200, resp.text


def test_wrong_token_returns_401(enable_auth, client):
    """An incorrect token is rejected."""
    resp = client.get(
        "/api/hpo_config?study_name=does_not_exist",
        headers={"X-HPO-Token": "wrong_token"},
    )
    assert resp.status_code == 401


def test_empty_token_header_returns_401(enable_auth, client):
    """An empty X-HPO-Token header is treated as missing token."""
    resp = client.get(
        "/api/hpo_config?study_name=does_not_exist",
        headers={"X-HPO-Token": ""},
    )
    assert resp.status_code == 401


def test_login_with_correct_token_sets_cookie(enable_auth, client):
    """POST /api/login with correct token returns a session cookie."""
    resp = client.post("/api/login", json={"token": TEST_TOKEN})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("success") is True
    assert data.get("auth_required") is True

    cookies = resp.headers.get("set-cookie", "")
    assert "hpo_session" in cookies
    assert "HttpOnly" in cookies


def test_login_with_wrong_token_returns_401(enable_auth, client):
    """POST /api/login with wrong token is rejected."""
    resp = client.post("/api/login", json={"token": "bogus"})
    assert resp.status_code == 401


def test_cookie_based_auth_passes_protected_route(enable_auth, client, initialized_study):
    """A request carrying the hpo_session cookie passes auth on protected routes."""
    login_resp = client.post("/api/login", json={"token": TEST_TOKEN})
    assert login_resp.status_code == 200
    cookie = login_resp.headers.get("set-cookie", "")
    assert "hpo_session" in cookie

    resp = client.get(
        f"/api/study_details?study_name={initialized_study}",
        headers={"Cookie": cookie},
    )
    assert resp.status_code == 200, resp.text


def test_login_returns_auth_required_false_when_no_token_configured(client):
    """When no HPO_SECRET_TOKEN is set, login reports auth is not required."""
    assert "HPO_SECRET_TOKEN" not in os.environ
    resp = client.post("/api/login", json={"token": "anything"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("auth_required") is False
