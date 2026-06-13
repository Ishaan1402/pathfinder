# tests/test_manifest.py — Tests for the manifest schema, validation rules, CLI integration, and DB roundtrips.

import os
import sys
import json
import yaml
import pytest
import subprocess
from src.manifest import (
    validate_manifest,
    _manifest_params_to_search_space,
    _manifest_to_hpo_config,
    ParamType,
    ObjectiveDirection
)
from src.db_manager import get_db_session
from src.schema import SystemConfiguration

@pytest.fixture
def base_manifest_data():
    return {
        "study_name": "segmentation_hpo_test",
        "metrics": {
            "primary_score": "dice",
            "objectives": [
                {
                    "name": "dice",
                    "direction": "maximize",
                    "label": "Dice Score"
                },
                {
                    "name": "loss",
                    "direction": "minimize",
                    "label": "BCE Loss"
                }
            ]
        },
        "params": [
            {
                "name": "learning_rate",
                "type": "float_log",
                "min": 1e-4,
                "max": 1e-1
            },
            {
                "name": "batch_size",
                "type": "categorical",
                "options": [16, 32, 64]
            },
            {
                "name": "optimizer",
                "type": "categorical",
                "options": ["adam", "sgd"]
            },
            {
                "name": "use_dropout",
                "type": "bool"
            },
            {
                "name": "num_epochs",
                "type": "fixed",
                "value": 15
            }
        ],
        "eval_protocol": {
            "enabled": True,
            "fixed_resolution": 512,
            "train_resolution_param": "resolution"
        },
        "worker": {
            "entrypoint": "python train.py --lr {learning_rate}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    }

def test_validate_manifest_success(base_manifest_data):
    errors, warnings = validate_manifest(base_manifest_data)
    assert len(errors) == 0
    assert len(warnings) == 0

def test_validate_manifest_errors(base_manifest_data):
    # Rule 1: study_name is empty/missing
    data = base_manifest_data.copy()
    data["study_name"] = ""
    errors, _ = validate_manifest(data)
    assert any("study_name is missing or empty" in e for e in errors)

    # Rule 2: metrics.objectives has at least one objective
    data = base_manifest_data.copy()
    data["metrics"] = {"primary_score": "dice", "objectives": []}
    errors, _ = validate_manifest(data)
    assert any("metrics.objectives must contain at least one objective definition" in e for e in errors)

    # Rule 3: primary_score references non-existing objective
    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "accuracy",
        "objectives": [{"name": "dice", "direction": "maximize", "label": "Dice"}]
    }
    errors, _ = validate_manifest(data)
    assert any("metrics.primary_score must reference a valid defined objective name" in e for e in errors)

    # Rule 4: Every objective has name, direction, label
    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "dice",
        "objectives": [{"name": "", "direction": "invalid_dir", "label": ""}]
    }
    errors, _ = validate_manifest(data)
    assert len(errors) > 0

    # Rule 5: Duplicate objective names
    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "dice",
        "objectives": [
            {"name": "dice", "direction": "maximize", "label": "Dice"},
            {"name": "dice", "direction": "minimize", "label": "Dice 2"}
        ]
    }
    errors, _ = validate_manifest(data)
    assert any("Duplicate objective name found" in e for e in errors)

    # Rule 6: params has at least one tunable parameter
    data = base_manifest_data.copy()
    data["params"] = [{"name": "epochs", "type": "fixed", "value": 10}]
    errors, _ = validate_manifest(data)
    assert any("params must contain at least one tunable parameter (non-fixed type)" in e for e in errors)

    # Rule 7: Invalid param type
    data = base_manifest_data.copy()
    data["params"] = [{"name": "lr", "type": "invalid_type"}]
    errors, _ = validate_manifest(data)
    assert any("has invalid type" in e for e in errors)

    # Rule 8: float/int bounds missing or min >= max
    data = base_manifest_data.copy()
    data["params"] = [{"name": "lr", "type": "float", "min": 0.5, "max": 0.1}]
    errors, _ = validate_manifest(data)
    assert any("min bound must be strictly less than max bound" in e for e in errors)

    # Rule 9: float_log min <= 0
    data = base_manifest_data.copy()
    data["params"] = [{"name": "lr", "type": "float_log", "min": 0.0, "max": 0.1}]
    errors, _ = validate_manifest(data)
    assert any("must have min bound > 0" in e for e in errors)

    # Rule 10: categorical options is non-empty list
    data = base_manifest_data.copy()
    data["params"] = [{"name": "batch_size", "type": "categorical", "options": []}]
    errors, _ = validate_manifest(data)
    assert any("must have a non-empty options list" in e for e in errors)

    # Rule 12: fixed params value is present
    data = base_manifest_data.copy()
    data["params"] = [{"name": "num_epochs", "type": "fixed"}]
    errors, _ = validate_manifest(data)
    assert any("must have a 'value' defined" in e for e in errors)

    # Rule 13: Duplicate parameter names
    data = base_manifest_data.copy()
    data["params"] = [
        {"name": "lr", "type": "float", "min": 0.1, "max": 0.5},
        {"name": "lr", "type": "float", "min": 0.2, "max": 0.6}
    ]
    errors, _ = validate_manifest(data)
    assert any("Duplicate parameter name found: 'lr'" in e for e in errors)

    # Rule 14: Conflicting reserved names
    data = base_manifest_data.copy()
    data["params"] = [{"name": "trial_id", "type": "float", "min": 1, "max": 10}]
    errors, _ = validate_manifest(data)
    assert any("conflicts with Pathfinder reserved names" in e for e in errors)

    # Rule 15: Support at most 2 objectives (one maximize, one minimize)
    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "acc",
        "objectives": [
            {"name": "acc", "direction": "maximize", "label": "Acc"},
            {"name": "prec", "direction": "maximize", "label": "Prec"},
            {"name": "loss", "direction": "minimize", "label": "Loss"}
        ]
    }
    errors, _ = validate_manifest(data)
    assert any("At most 2 objectives can be defined" in e for e in errors)

    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "acc",
        "objectives": [
            {"name": "acc", "direction": "maximize", "label": "Acc"},
            {"name": "prec", "direction": "maximize", "label": "Prec"}
        ]
    }
    errors, _ = validate_manifest(data)
    assert any("one must be 'maximize' and the other must be 'minimize'" in e for e in errors)

def test_validate_manifest_warnings(base_manifest_data):
    # Warning 1: Narrow bounds
    data = base_manifest_data.copy()
    data["params"] = [
        {"name": "learning_rate", "type": "float", "min": 0.100, "max": 0.1005}
    ]
    _, warnings = validate_manifest(data)
    assert any("Consider widening bounds" in w for w in warnings)

    # Warning 2: Wide log span
    data = base_manifest_data.copy()
    data["params"] = [
        {"name": "learning_rate", "type": "float_log", "min": 1e-8, "max": 10.0}
    ]
    _, warnings = validate_manifest(data)
    assert any("orders of magnitude" in w for w in warnings)

    # Warning 3: Single categorical option
    data = base_manifest_data.copy()
    data["params"] = [
        {"name": "batch_size", "type": "categorical", "options": [16]}
    ]
    _, warnings = validate_manifest(data)
    assert any("has only one option — should it be 'fixed'?" in w for w in warnings)

    # Warning 4: No eval protocol
    data = base_manifest_data.copy()
    data["eval_protocol"] = {"enabled": False}
    _, warnings = validate_manifest(data)
    assert any("Eval protocol is disabled" in w for w in warnings)

    # Warning 5: Single objective minimizing
    data = base_manifest_data.copy()
    data["metrics"] = {
        "primary_score": "loss",
        "objectives": [{"name": "loss", "direction": "minimize", "label": "BCE Loss"}]
    }
    _, warnings = validate_manifest(data)
    assert any("Only objective is minimize" in w for w in warnings)

    # Warning 6: Large parameter count (>10)
    data = base_manifest_data.copy()
    data["params"] = [{"name": f"param_{i}", "type": "float", "min": 0, "max": 1} for i in range(12)]
    _, warnings = validate_manifest(data)
    assert any("Large search space — consider starting with fewer parameters" in w for w in warnings)

    # Warning 7: Mixed naming styles
    data = base_manifest_data.copy()
    data["params"] = [
        {"name": "snake_case_name", "type": "float", "min": 0, "max": 1},
        {"name": "camelCaseName", "type": "float", "min": 0, "max": 1}
    ]
    _, warnings = validate_manifest(data)
    assert any("Parameter names use inconsistent styles" in w for w in warnings)

def test_mappings(base_manifest_data):
    space = _manifest_params_to_search_space(base_manifest_data["params"])
    assert "learning_rate" in space
    assert space["learning_rate"]["type"] == "float_log"
    assert space["batch_size"]["type"] == "categorical"
    assert space["use_dropout"]["type"] == "categorical"
    assert space["num_epochs"]["type"] == "categorical"
    assert space["num_epochs"]["options"] == [15]

    config = _manifest_to_hpo_config(base_manifest_data)
    assert config["config_version"] == 2
    assert config["metric_score_label"] == "Dice Score"
    assert config["metric_loss_label"] == "BCE Loss"
    assert config["eval_protocol"]["enabled"] is True
    assert config["eval_protocol"]["fixed_resolution"] == 512

def test_hpo_config_versioning():
    from src.hpo_config import load_hpo_config, save_hpo_config
    # Test fallback / default config yields version 2
    cfg_new = load_hpo_config("nonexistent_test_study_v2")
    assert cfg_new["config_version"] == 2
    assert cfg_new["metric_score_label"] == "Score"
    assert "legacy_param_aliases" not in cfg_new

    # Save a version 1 legacy configuration and verify it merges with legacy defaults
    legacy_cfg = {
        "config_version": 1,
        "metric_score_label": "Dice",
        "eval_protocol": {
            "enabled": True,
            "fixed_resolution": 256
        }
    }
    save_hpo_config(legacy_cfg, "legacy_test_study_v1")
    cfg_legacy = load_hpo_config("legacy_test_study_v1")
    assert cfg_legacy.get("config_version", 1) == 1
    assert cfg_legacy["metric_score_label"] == "Dice"
    assert cfg_legacy["metric_loss_label"] == "BCE"
    assert cfg_legacy["legacy_param_aliases"] == {"encoder_name": "model_capacity"}

def test_cli_validate_success(tmp_path, base_manifest_data):
    yaml_file = tmp_path / "manifest.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(base_manifest_data, f)

    res = subprocess.run(
        [sys.executable, "hpo_cli.py", "validate", str(yaml_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    assert "Manifest is valid" in res.stdout

def test_cli_validate_errors(tmp_path, base_manifest_data):
    data = base_manifest_data.copy()
    data["study_name"] = "" # Invalid
    yaml_file = tmp_path / "manifest.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

    res = subprocess.run(
        [sys.executable, "hpo_cli.py", "validate", str(yaml_file)],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "error(s) found" in res.stdout

def test_cli_init_and_manifest_roundtrip(tmp_path, base_manifest_data):
    study_name = "test_cli_manifest_study"
    data = base_manifest_data.copy()
    data["study_name"] = study_name
    yaml_file = tmp_path / "manifest.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

    # 1. Initialize study via CLI
    res = subprocess.run(
        [sys.executable, "hpo_cli.py", "init", str(yaml_file), "--force"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    assert "successfully initialized" in res.stdout.lower()

    # 2. Try to initialize again without force (should error out)
    res_err = subprocess.run(
        [sys.executable, "hpo_cli.py", "init", str(yaml_file)],
        capture_output=True,
        text=True
    )
    assert res_err.returncode != 0
    assert "already exists. Refusing to initialize" in res_err.stdout

    # 3. Export study via CLI manifest command
    res_manifest = subprocess.run(
        [sys.executable, "hpo_cli.py", "manifest", study_name],
        capture_output=True,
        text=True
    )
    assert res_manifest.returncode == 0
    exported_data = yaml.safe_load(res_manifest.stdout)
    assert exported_data["study_name"] == study_name
    assert len(exported_data["params"]) == len(data["params"])
    assert exported_data["metrics"]["primary_score"] == data["metrics"]["primary_score"]

def test_api_endpoints(client, base_manifest_data):
    study_name = "test_api_manifest_study"
    base_manifest_data["study_name"] = study_name

    # Validate endpoint
    res = client.post("/api/validate_manifest", json={"yaml": yaml.dump(base_manifest_data)})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True

    # Init endpoint
    res_init = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(base_manifest_data)})
    assert res_init.status_code == 200
    data_init = res_init.json()
    assert data_init["success"] is True
    assert data_init["study_name"] == study_name

    # Init duplicate error without force
    res_dup = client.post("/api/init_from_manifest?force=false", json={"yaml": yaml.dump(base_manifest_data)})
    assert res_dup.status_code == 400
    assert "already exists" in res_dup.json()["detail"]


def test_manifest_metric_ordering(client, base_manifest_data):
    import optuna
    from src.db_manager import get_db_session
    from src.schema import TrialResult
    
    # 1. Order [maximize, minimize] -> (Dice, Loss)
    study_name_1 = "test_order_max_min"
    data_1 = base_manifest_data.copy()
    data_1["study_name"] = study_name_1
    data_1["metrics"] = {
        "primary_score": "dice",
        "objectives": [
            {"name": "dice", "direction": "maximize", "label": "Dice"},
            {"name": "loss", "direction": "minimize", "label": "Loss"}
        ]
    }
    
    res_init_1 = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(data_1)})
    assert res_init_1.status_code == 200
    
    # Suggest trial
    res_sug_1 = client.post("/api/suggest_trial", json={"study_name": study_name_1, "worker_id": "w1"})
    assert res_sug_1.status_code == 200
    trial_id_1 = res_sug_1.json()["trial_id"]
    
    # Complete trial with dice=0.95, loss=0.05
    res_comp_1 = client.post("/api/complete_trial", json={
        "study_name": study_name_1, "trial_id": trial_id_1, "worker_id": "w1",
        "epoch": 1, "score": 0.95, "loss": 0.05, "weights_path": "m.pt",
        "history": [{"epoch": 1, "score": 0.95, "loss": 0.05, "dice": 0.95, "bce": 0.05}], "state": "COMPLETE",
    })
    assert res_comp_1.status_code == 200
    
    # Check Optuna trial values
    study_1 = optuna.load_study(study_name=study_name_1, storage=os.environ["HPO_DATABASE_URL"])
    trials_1 = study_1.get_trials()
    assert len(trials_1) == 1
    # Maximize is 1st (index 0), Minimize is 2nd (index 1)
    assert trials_1[0].values == [0.95, 0.05]

    # 2. Order [minimize, maximize] -> (Loss, Dice)
    study_name_2 = "test_order_min_max"
    data_2 = base_manifest_data.copy()
    data_2["study_name"] = study_name_2
    data_2["metrics"] = {
        "primary_score": "dice",
        "objectives": [
            {"name": "loss", "direction": "minimize", "label": "Loss"},
            {"name": "dice", "direction": "maximize", "label": "Dice"}
        ]
    }
    
    res_init_2 = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(data_2)})
    assert res_init_2.status_code == 200
    
    res_sug_2 = client.post("/api/suggest_trial", json={"study_name": study_name_2, "worker_id": "w2"})
    assert res_sug_2.status_code == 200
    trial_id_2 = res_sug_2.json()["trial_id"]
    
    res_comp_2 = client.post("/api/complete_trial", json={
        "study_name": study_name_2, "trial_id": trial_id_2, "worker_id": "w2",
        "epoch": 1, "score": 0.95, "loss": 0.05, "weights_path": "m.pt",
        "history": [{"epoch": 1, "score": 0.95, "loss": 0.05, "dice": 0.95, "bce": 0.05}], "state": "COMPLETE",
    })
    assert res_comp_2.status_code == 200
    
    study_2 = optuna.load_study(study_name=study_name_2, storage=os.environ["HPO_DATABASE_URL"])
    trials_2 = study_2.get_trials()
    assert len(trials_2) == 1
    # Minimize is 1st (index 0) -> loss, Maximize is 2nd (index 1) -> score
    assert trials_2[0].values == [0.05, 0.95]


def test_single_objective_minimize(client, base_manifest_data):
    import optuna
    study_name = "test_single_min"
    data = base_manifest_data.copy()
    data["study_name"] = study_name
    data["metrics"] = {
        "primary_score": "loss",
        "objectives": [
            {"name": "loss", "direction": "minimize", "label": "Loss"}
        ]
    }
    
    res_init = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(data)})
    assert res_init.status_code == 200
    
    study = optuna.load_study(study_name=study_name, storage=os.environ["HPO_DATABASE_URL"])
    assert len(study.directions) == 1
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    
    res_sug = client.post("/api/suggest_trial", json={"study_name": study_name, "worker_id": "w3"})
    assert res_sug.status_code == 200
    trial_id = res_sug.json()["trial_id"]
    
    res_comp = client.post("/api/complete_trial", json={
        "study_name": study_name, "trial_id": trial_id, "worker_id": "w3",
        "epoch": 1, "score": 0.0, "loss": 0.035, "weights_path": "m.pt",
        "history": [{"epoch": 1, "score": 0.0, "loss": 0.035, "dice": 0.0, "bce": 0.035}], "state": "COMPLETE",
    })
    assert res_comp.status_code == 200
    
    study = optuna.load_study(study_name=study_name, storage=os.environ["HPO_DATABASE_URL"])
    trials = study.get_trials()
    assert len(trials) == 1
    assert trials[0].value == 0.035


def test_deep_cleanup_on_force_overwrite(client, base_manifest_data):
    from src.db_manager import get_db_session
    from src.schema import SystemConfiguration, TrialResult
    
    study_name = "test_deep_cleanup_study"
    data = base_manifest_data.copy()
    data["study_name"] = study_name
    
    # 1. Initialize first time
    res_init1 = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(data)})
    assert res_init1.status_code == 200
    
    res_sug = client.post("/api/suggest_trial", json={"study_name": study_name, "worker_id": "w4"})
    assert res_sug.status_code == 200
    trial_id = res_sug.json()["trial_id"]
    
    res_comp = client.post("/api/complete_trial", json={
        "study_name": study_name, "trial_id": trial_id, "worker_id": "w4",
        "epoch": 1, "score": 0.8, "loss": 0.2, "weights_path": "m.pt",
        "history": [{"epoch": 1, "score": 0.8, "loss": 0.2, "dice": 0.8, "bce": 0.2}], "state": "COMPLETE",
    })
    assert res_comp.status_code == 200
    
    # Check that rows exist
    with get_db_session() as session:
        assert session.query(SystemConfiguration).filter_by(study_name=study_name).count() > 0
        assert session.query(TrialResult).filter_by(study_name=study_name).count() > 0
        
    # 2. Force re-initialize
    res_init2 = client.post("/api/init_from_manifest?force=true", json={"yaml": yaml.dump(data)})
    assert res_init2.status_code == 200
    
    # Check that previous TrialResult rows are completely cleaned up and only fresh config remains
    with get_db_session() as session:
        assert session.query(TrialResult).filter_by(study_name=study_name).count() == 0
        assert session.query(SystemConfiguration).filter_by(study_name=study_name).count() > 0


def test_manifest_validation_rules(base_manifest_data):
    from src.manifest import validate_manifest, _manifest_to_hpo_config
    
    # Test 1: Invalid rules types
    data = base_manifest_data.copy()
    data["validation_rules"] = {
        "enabled": "NOT_A_BOOL",
        "score_min": "NOT_A_NUMBER"
    }
    errors, warnings = validate_manifest(data)
    assert any("validation_rules.enabled must be a boolean" in e for e in errors)
    assert any("validation_rules.score_min must be a numeric value" in e for e in errors)
    
    # Test 2: Valid rules success
    data["validation_rules"] = {
        "enabled": True,
        "score_min": 0.1,
        "loss_min": 0.05,
        "max_epoch_jump": 0.2
    }
    errors, warnings = validate_manifest(data)
    assert len(errors) == 0
    
    # Test 3: config mapping
    config = _manifest_to_hpo_config(data)
    rules = config.get("validation_rules")
    assert rules is not None
    assert rules["enabled"] is True
    assert rules["score_min"] == 0.1
    assert rules["loss_min"] == 0.05
    assert rules["max_epoch_jump"] == 0.2

