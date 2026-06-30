import optuna
import json
import os
import datetime
from sqlalchemy.orm import Session
from src.db_manager import engine, SessionLocal
from src.schema import TrialResult

def extract_study_data():
    study_name = "bridge_crack_study"
    print(f"Loading Optuna study '{study_name}'...")
    
    # 1. Load the Optuna study
    db_url = os.getenv("HPO_DATABASE_URL", "sqlite:///hpo_studies.db")
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
    except KeyError:
        print(f"Error: Study '{study_name}' not found in database.")
        return
        
    print(f"Loading Pathfinder-specific trial results for '{study_name}'...")
    # 2. Query Pathfinder TrialResult metadata from the DB
    pathfinder_results = {}
    with SessionLocal() as session:
        results = session.query(TrialResult).filter_by(study_name=study_name).all()
        for r in results:
            pathfinder_results[r.trial_id] = r.to_dict()
            
    print(f"Aggregating {len(study.trials)} trials...")
    # 3. Combine Optuna trials with Pathfinder trial results
    all_trials_data = []
    
    for t in study.trials:
        # Get basic Optuna trial info
        trial_number = t.number
        trial_id = t._trial_id
        state = t.state.name
        params = t.params
        datetime_start = t.datetime_start.isoformat() if t.datetime_start else None
        datetime_complete = t.datetime_complete.isoformat() if t.datetime_complete else None
        duration = t.duration.total_seconds() if t.duration else None
        values = t.values  # This is a list/tuple of objective values
        
        # Primary score and loss from values or parameters
        # In multi-objective: direction minimize (loss/obj 0), maximize (score/obj 1)
        optuna_loss = values[0] if values and len(values) > 0 else None
        optuna_score = values[1] if values and len(values) > 1 else None
        
        # Get Pathfinder metadata if available using _trial_id
        pf_info = pathfinder_results.get(trial_id, {})
        
        # Merge data
        trial_record = {
            "trial_number": trial_number,
            "trial_id": trial_id,
            "state": state,
            "params": params,
            "datetime_start": datetime_start,
            "datetime_complete": datetime_complete,
            "duration_seconds": duration,
            "optuna_loss": optuna_loss,
            "optuna_score": optuna_score,
            "epoch_reached": pf_info.get("epoch_reached"),
            "primary_score": pf_info.get("primary_score"),
            "primary_loss": pf_info.get("primary_loss"),
            "oom_triggered": pf_info.get("oom_triggered"),
            "failure_tag": pf_info.get("failure_tag"),
            "gpu_model": pf_info.get("gpu_model"),
            "max_vram_gb": pf_info.get("max_vram_gb"),
            "health_tier": pf_info.get("health_tier"),
            "health_reason": pf_info.get("health_reason"),
            "git_commit": pf_info.get("git_commit"),
            "dataset_version": pf_info.get("dataset_version"),
        }
        all_trials_data.append(trial_record)
        
    # Sort by trial_number
    all_trials_data.sort(key=lambda x: x["trial_number"])
    
    # Save to JSON
    output_path = "bridge_crack_study_trials.json"
    with open(output_path, "w") as f:
        json.dump(all_trials_data, f, indent=2)
    print(f"Successfully saved all trials data to '{output_path}'.")
    
    # Filter completed trials to show progress
    completed_trials = [t for t in all_trials_data if t["state"] == "COMPLETE"]
    print(f"Total trials: {len(all_trials_data)}")
    print(f"Completed trials: {len(completed_trials)}")
    
    if not completed_trials:
        print("No completed trials found to evaluate improvement.")
        return
        
    # Find start trials vs best trials
    first_completed = completed_trials[:3]
    best_completed = sorted(completed_trials, key=lambda x: x["primary_score"] or 0, reverse=True)[:3]
    
    print("\n--- FIRST COMPLETED TRIALS ---")
    for t in first_completed:
        print(f"Trial #{t['trial_number']} (ID={t['trial_id']}): Score={t['primary_score']}, Loss={t['primary_loss']}, Params={t['params']}")
        
    print("\n--- BEST COMPLETED TRIALS ---")
    for t in best_completed:
        print(f"Trial #{t['trial_number']} (ID={t['trial_id']}): Score={t['primary_score']}, Loss={t['primary_loss']}, Params={t['params']}")
        
    initial_score = first_completed[0]['primary_score'] if first_completed else None
    best_score = best_completed[0]['primary_score'] if best_completed else None
    
    if initial_score is not None and best_score is not None:
        diff = best_score - initial_score
        print(f"\nImprovement in Best Score: {initial_score} -> {best_score} (Gain: +{diff:.6f})")
    else:
        print("\nCould not calculate improvement due to missing scores.")

if __name__ == "__main__":
    extract_study_data()
