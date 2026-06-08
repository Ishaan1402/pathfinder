import os
import json
import sys
import threading
import time
import requests
import uvicorn
import subprocess

# 1. Clean old test database if it exists to start fresh (before database engine initialization)
if __name__ == "__main__":
    db_file = "test_hpo_studies.db"
    if os.path.exists(db_file):
        print(f"Removing existing test database: {db_file}")
        try:
            os.remove(db_file)
        except OSError as e:
            print(f"Warning: Could not remove db file: {e}")

    # Override database URL to point to a test SQLite database before imports
    os.environ["HPO_DATABASE_URL"] = f"sqlite:///{db_file}"

    # Make sure workspace is in python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Start broker in background thread
    broker_port = 8123
    os.environ["HPO_BROKER_URL"] = f"http://127.0.0.1:{broker_port}"

    def start_broker():
        from broker import app
        uvicorn.run(app, host="127.0.0.1", port=broker_port, log_level="warning")

    print("Starting HTTP broker in background thread...")
    broker_thread = threading.Thread(target=start_broker, daemon=True)
    broker_thread.start()
    time.sleep(2.0)  # Wait for uvicorn to bind and start

    # Import dependencies after environment setup
    from src.db_manager import init_db, get_db_session
    from src.schema import TrialResult, SystemConfiguration, StudyStatus, StudyCard
    from hpo_mcp_server import (
        initialize_study,
        get_study_data,
        validate_search_space,
        update_search_space,
        generate_model_card,
        submit_agent_review
    )
    from simulators.training_worker import run_training_worker



def run_integration_test():
    print("==================================================")
    print("STARTING PATHFINDER INTEGRATION TEST (DB-BACKED)")
    print("==================================================\n")

    # Reinitialize DB tables
    print("Initializing SQLite database tables...")
    init_db()
    
    study_name = "unet_crack_segmentation_test"
    active_search_space = {
        "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float_log"},
        "batch_size": {"options": [2, 4, 8, 16, 32, 64], "active": [2, 4, 8, 16, 32, 64], "type": "categorical"},
        "resolution": {"options": [256, 512, 1024], "active": [256, 512, 1024], "type": "categorical"},
        "model_capacity": {"options": ["narrow", "wide"], "active": ["narrow", "wide"], "type": "categorical"},
        "loss_weight_ratio": {"min": 0.0, "max": 1.0, "type": "float"}
    }
    hpo_config = {
        "eval_protocol": {
            "enabled": True,
            "fixed_resolution": 512,
            "train_resolution_param": "resolution",
            "fixed_dice_attr": "dice_eval_fixed",
            "fixed_bce_attr": "bce_eval_fixed"
        },
        "metric_score_label": "Dice",
        "metric_loss_label": "BCE"
    }
    project_context = {
        "hypothesis": "Testing U-Net segmentation models on crack images.",
        "gpu_model": "NVIDIA L4",
        "gpu_capacity_gb": 24.0
    }

    # 1. Test Study Initialization
    print("\n--- [Step 1: Initializing Study in Database] ---")
    init_msg = initialize_study(
        study_name=study_name,
        active_search_space=active_search_space,
        hpo_config=hpo_config,
        project_context=project_context,
        multi_objective=True
    )
    print(init_msg)
    
    # Verify records were inserted in SystemConfiguration
    with get_db_session() as session:
        space_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        assert space_row is not None, "Active search space should be stored in system_configuration!"
        
        status_row = session.query(StudyStatus).filter_by(study_name=study_name).first()
        assert status_row is not None, "Study status should be initialized!"
        assert status_row.health_tier == "healthy", "Initial health tier should be healthy!"

    # 2. Test Search Space Pre-flight Validation
    print("\n--- [Step 2: Verifying Search Space Validation] ---")
    # A. Propose a valid space config
    valid_val = validate_search_space(active_search_space)
    print(f"Valid validation response: {valid_val}")
    assert valid_val["valid"], "Search space configuration should be valid!"
    
    # B. Propose an invalid space config (min >= max)
    invalid_space = {
        "learning_rate": {"min": 1e-2, "max": 1e-5, "type": "float_log"}
    }
    invalid_val = validate_search_space(invalid_space)
    print(f"Invalid validation response: {invalid_val}")
    assert not invalid_val["valid"], "Search space validation should fail for min >= max!"
    assert len(invalid_val["errors"]) > 0, "Errors should be reported!"

    # 2.5 Run a quick mock trial to satisfy trials_evaluated > 0 rule
    print("\n--- [Step 2.5: Running a quick mock trial to satisfy trials_evaluated > 0] ---")
    from src.hpo_client import TrialSession
    session = TrialSession(broker_url=f"http://127.0.0.1:{broker_port}", study_name=study_name)
    trial_data = session.suggest()
    session.complete(epoch=0, score=0.5, loss=0.5)

    # 3. Test Manual Parameter Suggestion & Guardrails
    print("\n--- [Step 3: Suggesting Next Trial with Manual Parameters] ---")
    # A. Propose invalid resolution (not multiple of 32)
    invalid_params_1 = {
        "learning_rate": 1e-3,
        "batch_size": 16,
        "resolution": 500,  # Invalid
        "model_capacity": "narrow",
        "loss_weight_ratio": 0.5
    }
    print(f"Proposing invalid parameters (resolution 500): {invalid_params_1}")
    res_1 = submit_agent_review(
        study_name=study_name,
        summary="Testing invalid resolution boundary",
        health_rating=3,
        policy_action="enqueue_one_manual_trial",
        model_version="coordinator",
        prompt_strategy="test_strategy",
        estimated_score_improvement=-1.0,
        cited_best_trial=0,
        manual_trial=invalid_params_1,
        force=True
    )
    print(f"Response: {res_1}\n")
    assert not res_1["success"], "Should have failed due to resolution constraints!"

    # B. Propose valid manual parameters
    valid_manual = {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "resolution": 256,
        "model_capacity": "narrow",
        "loss_weight_ratio": 0.3
    }
    print(f"Proposing valid manual parameters: {valid_manual}")
    res_valid = submit_agent_review(
        study_name=study_name,
        summary="Starting with a reasonable base configuration",
        health_rating=4,
        policy_action="enqueue_one_manual_trial",
        model_version="coordinator",
        prompt_strategy="test_strategy",
        estimated_score_improvement=0.05,
        cited_best_trial=0,
        manual_trial=valid_manual,
        force=True
    )
    print(f"Response: {res_valid}")
    assert res_valid["success"], f"Should have successfully enqueued: {res_valid.get('error')}"

    # 4. Simulate Training Worker trials
    print("\n--- [Step 4: Running Decentralized Training Worker Simulation via HTTP] ---")
    run_training_worker(
        study_name=study_name,
        agent_model="gemini-3.5-flash",
        prompt_strategy="tpe_guided_v1",
        max_trials=7,
        epochs_per_trial=5,
        broker_url=f"http://127.0.0.1:{broker_port}"
    )

    # 5. Fetch Study Data Compacted Packet
    print("\n--- [Step 5: Fetching Compacted Review Packet] ---")
    packet = get_study_data(study_name=study_name)
    print(f"Compacted Packet structure keys: {list(packet.keys())}")
    assert "trial_bins" in packet, "Packet must contain binned trials."
    assert "fanova_importances" in packet, "Packet must contain parameter importances."
    assert "spearman_correlations" in packet, "Packet must contain Spearman correlations."
    assert "vram_telemetry" in packet, "Packet must contain VRAM telemetry."
    
    print(f"Elite Trials count: {len(packet['trial_bins']['elite'])}")
    print(f"Noise floor trials summary: {packet['trial_bins']['noise_floor']['count']} trials, median score={packet['trial_bins']['noise_floor']['median_score']:.4f}")
    print(f"Failure combinations matrix: {packet['trial_bins']['failure_matrix']}")
    print(f"fANOVA Importances: {packet['fanova_importances']}")
    print(f"VRAM Telemetry details: GPU={packet['vram_telemetry']['gpu_model']}, OOM count={packet['vram_telemetry']['oom_count']}")

    # 6. Test Proposing and Applying Search Space Updates
    print("\n--- [Step 6: Proposing and Applying Search Space Updates] ---")
    proposal = {
        "learning_rate": {"min": 1e-4, "max": 1e-3}
    }
    print(f"Proposing search space update: {proposal}")
    prop_msg = update_search_space(study_name=study_name, space_config=proposal, apply=False)
    print(prop_msg)
    
    # Check pending changes row
    with get_db_session() as session:
        row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        assert row is not None, "Pending changes should be written to SQLite!"
        
    print("Applying pending search space changes...")
    apply_msg = update_search_space(study_name=study_name, space_config=proposal, apply=True)
    print(apply_msg)
    
    # Verify change is applied and pending is cleared
    with get_db_session() as session:
        pending_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        assert pending_row is None, "Pending changes should be deleted after apply!"
        
        space_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        current_space = json.loads(space_row.config_value)
        assert current_space["learning_rate"]["min"] == 1e-4, "Min learning rate should be updated to 1e-4!"
        assert current_space["learning_rate"]["max"] == 1e-3, "Max learning rate should be updated to 1e-3!"

    # 7. Test Synthesis / Generating Model Card
    print("\n--- [Step 7: Generating End-of-Study Model Card] ---")
    card_res = generate_model_card(study_name=study_name)
    print(card_res)
    assert card_res["success"], f"Failed to generate model card: {card_res.get('error')}"
    
    # Verify card indexed in DB and file exists
    assert os.path.exists(card_res["file_path"]), "Model card file should be written to disk!"
    with get_db_session() as session:
        card_row = session.query(StudyCard).filter_by(study_name=study_name).first()
        assert card_row is not None, "Model card should be indexed in SQLite database!"
        print(f"Indexed card metadata: {json.loads(card_row.metadata_json)}")

    # 8. Test Retrieving Study Cards (Querying Model Card)
    print("\n--- [Step 8: Querying Indexed Study Cards] ---")
    from hpo_mcp_server import get_study_cards
    cards = get_study_cards(study_name=study_name)
    print(f"Retrieved {len(cards)} card(s) from database.")
    assert len(cards) > 0, "Should have retrieved at least one study card!"
    assert cards[0]["markdown_content"].startswith("# Study Model Card:"), "Markdown content should contain generated model card!"
    
    # Test HTTP endpoint for study cards
    resp = requests.get(f"http://127.0.0.1:{broker_port}/api/study_cards?study_name={study_name}")
    resp_data = resp.json()
    assert resp_data["success"], "HTTP api/study_cards request should be successful!"
    assert len(resp_data["cards"]) > 0, "HTTP response should contain study cards!"
    print("Study cards query tests passed successfully!")

    # 9. Test Nudge Dismissal
    print("\n--- [Step 9: Testing Nudge Dismissal Persistence] ---")
    from src.hpo_config import load_hpo_config
    from broker import get_or_create_study
    from src.hpo_coordinator import study_eval_insights, compute_review_heuristics
    hpo_config = load_hpo_config(study_name)
    study = get_or_create_study(study_name)
    insights = study_eval_insights(study, hpo_config)
    heuristics = compute_review_heuristics(study, insights, hpo_config, study_name)
    
    # Dismiss nudge via HTTP API
    resp = requests.post(f"http://127.0.0.1:{broker_port}/api/dismiss_coordinator_nudge?study_name={study_name}")
    assert resp.status_code == 200, "Dismiss nudge endpoint should return 200"
    assert resp.json()["success"], "Dismiss nudge request should succeed"
    
    # Re-evaluate heuristics and verify dismissal is respected
    heuristics_after = compute_review_heuristics(study, insights, hpo_config, study_name)
    assert heuristics_after["already_dismissed"] == True, "already_dismissed should be True after dismissal!"
    assert heuristics_after["review_recommended"] == False, "review_recommended should be False after dismissal!"
    print("Nudge dismissal persistence tests passed successfully!")

    # 10. Test HPO_SECRET_TOKEN Authentication
    print("\n--- [Step 10: Testing HPO_SECRET_TOKEN Authentication] ---")
    os.environ["HPO_SECRET_TOKEN"] = "test_integration_token_123"
    
    resp_no_token = requests.get(f"http://127.0.0.1:{broker_port}/api/study_details?study_name={study_name}")
    assert resp_no_token.status_code == 401, "Request without token should fail with 401!"
    
    resp_bad_token = requests.get(f"http://127.0.0.1:{broker_port}/api/study_details?study_name={study_name}", headers={"X-HPO-Token": "bad_token"})
    assert resp_bad_token.status_code == 401, "Request with bad token should fail with 401!"
    
    resp_good_token = requests.get(f"http://127.0.0.1:{broker_port}/api/study_details?study_name={study_name}", headers={"X-HPO-Token": "test_integration_token_123"})
    assert resp_good_token.status_code == 200, "Request with correct header token should succeed!"
    
    resp_auth_token = requests.get(
        f"http://127.0.0.1:{broker_port}/api/study_details?study_name={study_name}",
        headers={"Authorization": "Bearer test_integration_token_123"}
    )
    assert resp_auth_token.status_code == 200, "Request with correct Authorization token should succeed!"

    del os.environ["HPO_SECRET_TOKEN"]
    print("HPO_SECRET_TOKEN middleware authentication tests passed successfully!")

    # 11. Test CLI Commands
    print("\n--- [Step 11: Testing CLI Commands] ---")
    
    # Test 'python hpo_cli.py status'
    cmd_status = subprocess.run(
        [sys.executable, "hpo_cli.py", "status", "--study", study_name],
        capture_output=True,
        text=True
    )
    assert cmd_status.returncode == 0, "hpo_cli.py status command failed!"
    assert "STUDY STATUS" in cmd_status.stdout, "CLI status output should contain study status header!"
    
    # Stage proposed changes manually
    with get_db_session() as session:
        session.merge(SystemConfiguration(
            study_name=study_name,
            config_key="pending_search_space",
            config_value=json.dumps({"learning_rate": {"min": 5e-5, "max": 5e-4}})
        ))
        session.commit()
        
    cmd_status_pending = subprocess.run(
        [sys.executable, "hpo_cli.py", "status", "--study", study_name],
        capture_output=True,
        text=True
    )
    assert "Pending Changes:   YES" in cmd_status_pending.stdout, "CLI status should report pending changes!"
    
    cmd_apply = subprocess.run(
        [sys.executable, "hpo_cli.py", "apply", "--study", study_name],
        capture_output=True,
        text=True
    )
    assert cmd_apply.returncode == 0, "hpo_cli.py apply command failed!"
    assert "Pending search space changes committed successfully." in cmd_apply.stdout, "CLI apply message missing!"
    
    with get_db_session() as session:
        space_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="active_search_space"
        ).first()
        current_space = json.loads(space_row.config_value)
        assert current_space["learning_rate"]["min"] == 5e-5, "CLI apply did not update active search space min learning rate!"
        assert current_space["learning_rate"]["max"] == 5e-4, "CLI apply did not update active search space max learning rate!"
        
    print("CLI commands integration tests passed successfully!")

    print("\n==================================================")
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_integration_test()
