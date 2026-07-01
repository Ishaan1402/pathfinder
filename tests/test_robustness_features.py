import os
import tempfile
import json
import pytest
import sqlite3
from src.db_manager import get_db_session
from src.schema import TrialResult
import hpo_cli

def test_zero_metric_rejection(client, initialized_study):
    # Suggest a trial:
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "test_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    # Complete with score=0.0, loss=0.0 (empty history triggers multi-objective rejection)
    payload = {
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "test_worker",
        "epoch": 2,
        "score": 0.0,
        "loss": 0.0,
        "weights_path": "model.pt",
        "history": [],
        "state": "COMPLETE"
    }
    resp = client.post("/api/complete_trial", json=payload)
    assert resp.status_code == 400
    assert "Likely training did not run" in resp.json()["detail"]

def test_nan_inf_metric_rejections(client, initialized_study):
    # Suggest a trial:
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "test_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    # Complete with score="NaN"
    payload = {
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "test_worker",
        "epoch": 2,
        "score": "NaN",
        "loss": 0.3,
        "weights_path": "model.pt",
        "history": [{"epoch": 2, "score": 0.0, "loss": 0.3}],
        "state": "COMPLETE"
    }
    
    resp = client.post("/api/complete_trial", json=payload)
    assert resp.status_code == 400
    assert "is NaN or Inf" in resp.json()["detail"]

    payload["score"] = "Infinity"
    resp = client.post("/api/complete_trial", json=payload)
    assert resp.status_code == 400
    assert "is NaN or Inf" in resp.json()["detail"]

def test_cli_export_import_roundtrip(client, initialized_study):
    # 1. Add some trials to the initialized study
    worker_id = "worker_1"
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": worker_id})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    # Report epoch to create history
    resp = client.post("/api/report_epoch", json={
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": worker_id,
        "epoch": 0,
        "score": 0.2,
        "loss": 0.8
    })
    assert resp.status_code == 200
    
    # Complete the trial
    resp = client.post("/api/complete_trial", json={
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": worker_id,
        "epoch": 0,
        "score": 0.5,
        "loss": 0.5,
        "weights_path": "model.pt",
        "history": [{"epoch": 0, "score": 0.5, "loss": 0.5}],
        "state": "COMPLETE",
        "python_version": "3.10",
        "cuda_version": "11.8",
        "pip_freeze": "torch==2.0.0",
        "platform": "linux",
        "hostname": "test-host",
        "git_commit": "abcdef12",
        "dataset_version": "v1"
    })
    assert resp.status_code == 200
    
    # 2. Export the study to a temporary file
    temp_dir = tempfile.mkdtemp()
    export_path = os.path.join(temp_dir, "export.json")
    
    class Args:
        study = initialized_study
        format = "json"
        output = export_path
        
    try:
        hpo_cli.cmd_export(Args())
    except SystemExit as e:
        assert e.code == 0
    
    assert os.path.exists(export_path)
    with open(export_path, "r") as f:
        data = json.load(f)
        assert data["study_name"] == initialized_study
        assert len(data["trials"]) == 1
        assert len(data["trial_results"]) == 1
        
    # 3. Import the study under a new name
    imported_study_name = f"{initialized_study}_imported"
    class ImportArgs:
        file = export_path
        rename = imported_study_name
        force = True
        
    try:
        hpo_cli.cmd_import(ImportArgs())
    except SystemExit as e:
        assert e.code == 0
    
    # 4. Verify import correctness
    resp = client.get(f"/api/study_details?study_name={imported_study_name}")
    assert resp.status_code == 200
    details = resp.json()
    assert details["study_name"] == imported_study_name
    assert len(details["trials"]) == 1
    t = details["trials"][0]
    assert t["state"] == "COMPLETE"
    assert t["git_commit"] == "abcdef12"
    assert t["dataset_version"] == "v1"
    
    # Clean up
    try:
        os.remove(export_path)
        os.rmdir(temp_dir)
    except Exception:
        pass

def test_cli_backup_command():
    temp_dir = tempfile.mkdtemp()
    backup_path = os.path.join(temp_dir, "test_backup.db")
    
    class BackupArgs:
        output = backup_path
        
    try:
        hpo_cli.cmd_backup(BackupArgs())
    except SystemExit as e:
        assert e.code == 0
    
    assert os.path.exists(backup_path)
    conn = sqlite3.connect(backup_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    assert "trial_results" in tables
    conn.close()
    
    try:
        os.remove(backup_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


def test_flat_env_fields_api(client, initialized_study):
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "env_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    payload = {
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "env_worker",
        "epoch": 1,
        "score": 0.6,
        "loss": 0.4,
        "weights_path": "model.pt",
        "history": [{"epoch": 1, "score": 0.6, "loss": 0.4}],
        "state": "COMPLETE",
        "python_version": "3.11.2",
        "cuda_version": "12.1",
        "pip_freeze": "torch==2.1.0\noptuna==3.5.0",
        "platform": "Darwin-22.3.0",
        "hostname": "macbook-pro",
        "git_commit": "1234567890abcdef1234567890abcdef12345678",
        "dataset_version": "mnist_v2"
    }
    
    resp = client.post("/api/complete_trial", json=payload)
    assert resp.status_code == 200
    
    resp = client.get(f"/api/study_details?study_name={initialized_study}")
    assert resp.status_code == 200
    details = resp.json()
    assert len(details["trials"]) > 0
    t = [x for x in details["trials"] if x["trial_id"] == trial_id][0]
    assert t["git_commit"] == "1234567890abcdef1234567890abcdef12345678"
    assert t["dataset_version"] == "mnist_v2"
    assert t["python_version"] == "3.11.2"
    assert t["cuda_version"] == "12.1"
    assert t["pip_freeze"] == "torch==2.1.0\noptuna==3.5.0"
    assert t["platform"] == "Darwin-22.3.0"
    assert t["hostname"] == "macbook-pro"


def test_transient_health_warning_clears(client, initialized_study):
    # Enable validation rules
    from src.hpo_config import save_hpo_config, load_hpo_config
    config = load_hpo_config(initialized_study)
    config["validation_rules"] = {
        "score_min": 0.0,
        "loss_min": 0.0,
        "max_epoch_jump": 0.5,
        "enabled": True
    }
    save_hpo_config(config, initialized_study)

    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "health_worker"})
    assert resp.status_code == 200
    trial_id_1 = resp.json()["trial_id"]
    
    # Report epoch with negative score to trigger warning (since score_min is 0.0)
    resp = client.post("/api/report_epoch", json={
        "study_name": initialized_study,
        "trial_id": trial_id_1,
        "worker_id": "health_worker",
        "epoch": 6,
        "score": -0.5,
        "loss": 0.5,
        "history": [{"epoch": 6, "score": -0.5, "loss": 0.5}]
    })
    assert resp.status_code == 200
    
    # Check study health - should be watch
    resp = client.get(f"/api/study_details?study_name={initialized_study}")
    assert resp.status_code == 200
    details = resp.json()
    assert details["health"]["tier"] == "watch"
    
    # Complete a healthy trial
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "health_worker"})
    assert resp.status_code == 200
    trial_id_2 = resp.json()["trial_id"]
    
    resp = client.post("/api/complete_trial", json={
        "study_name": initialized_study,
        "trial_id": trial_id_2,
        "worker_id": "health_worker",
        "epoch": 2,
        "score": 0.8,
        "loss": 0.2,
        "weights_path": "model.pt",
        "history": [{"epoch": 2, "score": 0.8, "loss": 0.2}],
        "state": "COMPLETE"
    })
    assert resp.status_code == 200
    
    # Complete the first warning trial as FAIL so it's not running
    resp = client.post("/api/complete_trial", json={
        "study_name": initialized_study,
        "trial_id": trial_id_1,
        "worker_id": "health_worker",
        "epoch": 6,
        "score": -0.5,
        "loss": 0.5,
        "history": [],
        "weights_path": "",
        "state": "FAIL"
    })
    assert resp.status_code == 200
    
    # Check study health - transient warning should be cleared
    resp = client.get(f"/api/study_details?study_name={initialized_study}")
    assert resp.status_code == 200
    details = resp.json()
    assert details["health"]["tier"] == "healthy"


def test_unbounded_metric_study_skips_validation(client, initialized_study):
    from src.hpo_config import save_hpo_config, load_hpo_config
    config = load_hpo_config(initialized_study)
    config["validation_rules"] = {"enabled": False}
    save_hpo_config(config, initialized_study)
    
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "unbounded_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    resp = client.post("/api/report_epoch", json={
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "unbounded_worker",
        "epoch": 6,
        "score": -0.5,
        "loss": 0.5,
        "history": [{"epoch": 6, "score": -0.5, "loss": 0.5}]
    })
    assert resp.status_code == 200
    
    resp = client.get(f"/api/study_details?study_name={initialized_study}")
    assert resp.status_code == 200
    details = resp.json()
    assert details["health"]["tier"] == "healthy"


def test_complete_partial_metrics(client, initialized_study):
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "partial_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    
    # Complete with score=0.0 and loss=0.5 should SUCCEED
    resp = client.post("/api/complete_trial", json={
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "partial_worker",
        "epoch": 2,
        "score": 0.0,
        "loss": 0.5,
        "weights_path": "model.pt",
        "history": [{"epoch": 2, "score": 0.0, "loss": 0.5}],
        "state": "COMPLETE"
    })
    assert resp.status_code == 200


def test_import_rollback_on_failure(client, initialized_study):
    """Failed imports must leave the database in a clean state.
    
    We export a valid study, corrupt the JSON so that the custom-table
    insertion blows up, then assert that the target study name was
    fully rolled back: no Optuna study and no trial_results rows.
    """
    import tempfile
    import optuna

    # 1. Add a trial so the export is non-trivial.
    resp = client.post("/api/suggest_trial", json={"study_name": initialized_study, "worker_id": "rollback_worker"})
    assert resp.status_code == 200
    trial_id = resp.json()["trial_id"]
    resp = client.post("/api/complete_trial", json={
        "study_name": initialized_study,
        "trial_id": trial_id,
        "worker_id": "rollback_worker",
        "epoch": 1,
        "score": 0.7,
        "loss": 0.3,
        "weights_path": "model.pt",
        "history": [{"epoch": 1, "score": 0.7, "loss": 0.3}],
        "state": "COMPLETE",
    })
    assert resp.status_code == 200

    # 2. Export to a temp file.
    temp_dir = tempfile.mkdtemp()
    export_path = os.path.join(temp_dir, "corrupt_export.json")

    class ExportArgs:
        study = initialized_study
        format = "json"
        output = export_path

    try:
        hpo_cli.cmd_export(ExportArgs())
    except SystemExit as e:
        assert e.code == 0

    # 3. Corrupt the exported JSON: inject an unparseable datetime into
    #    trial_results so the insertion raises an exception mid-import.
    with open(export_path, "r") as f:
        data = json.load(f)

    target_study = f"{initialized_study}_rollback_target"
    if data.get("trial_results"):
        data["trial_results"][0]["created_at"] = "NOT_A_VALID_DATETIME"

    with open(export_path, "w") as f:
        json.dump(data, f)

    # 4. Attempt the import — it must fail.
    class ImportArgs:
        file = export_path
        rename = target_study
        force = True

    with pytest.raises(SystemExit) as exc_info:
        hpo_cli.cmd_import(ImportArgs())
    assert exc_info.value.code != 0

    # 5. Assert no Optuna study was left behind.
    from src.db_manager import DATABASE_URL as _DB_URL
    all_study_names = [s.study_name for s in optuna.get_all_study_summaries(storage=_DB_URL)]
    assert target_study not in all_study_names, (
        f"Import rollback failed: Optuna study '{target_study}' still exists after a failed import."
    )

    # 6. Assert no trial_results rows were written.
    with get_db_session() as session:
        rows = session.query(TrialResult).filter_by(study_name=target_study).all()
        assert len(rows) == 0, (
            f"Import rollback failed: {len(rows)} trial_results rows orphaned for '{target_study}'."
        )

    # Cleanup.
    try:
        os.remove(export_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
