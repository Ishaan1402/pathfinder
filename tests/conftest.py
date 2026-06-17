"""Shared pytest fixtures and test-database setup.

This consolidates the per-module boilerplate that used to be copy-pasted at the top of
every test file: it points HPO_DATABASE_URL at a throwaway SQLite file BEFORE any
``src.*`` import binds the SQLAlchemy engine, and exposes reusable fixtures.
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path

# 1. Project root on sys.path.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 2. Bind a temp test DB before anything imports src.db_manager (which builds the
#    engine at import time). Idempotent so direct ``python -m unittest`` runs still work.
if "HPO_DATABASE_URL" not in os.environ:
    _fd, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="hpo_pytest_")
    os.close(_fd)
    test_db = Path(_TEST_DB_PATH).resolve().as_posix()
    os.environ["HPO_DATABASE_URL"] = f"sqlite:///{test_db}"

    import atexit

    @atexit.register
    def _cleanup_test_db():
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(_TEST_DB_PATH + suffix)
            except OSError:
                pass

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_database():
    """Create all tables once for the test session."""
    from src.db_manager import init_db
    init_db()


@pytest.fixture
def client():
    """FastAPI TestClient against the broker app."""
    from fastapi.testclient import TestClient
    from broker import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def unique_study_name():
    return f"test_http_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def initialized_study(unique_study_name):
    """Initialize a study (Optuna + config in DB) and return its name."""
    from hpo_mcp_server import initialize_study

    active_search_space = {
        "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float_log"},
        "batch_size": {"options": [2, 4, 8, 16], "active": [2, 4, 8, 16], "type": "categorical"},
        "loss_weight_ratio": {"min": 0.0, "max": 1.0, "type": "float"},
    }
    hpo_config = {
        "eval_protocol": {"enabled": False},
        "metric_score_label": "Score",
        "metric_loss_label": "Loss",
    }
    project_context = {"hypothesis": "pytest study", "gpu_model": "CPU", "gpu_capacity_gb": 8.0}
    initialize_study(
        study_name=unique_study_name,
        active_search_space=active_search_space,
        hpo_config=hpo_config,
        project_context=project_context,
        multi_objective=True,
    )
    return unique_study_name
