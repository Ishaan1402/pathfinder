import optuna
import csv
import os
from sqlalchemy.orm import Session
from src.db_manager import SessionLocal
from src.schema import TrialResult

def extract_filtered_csv():
    study_name = "bridge_crack_study"
    print(f"Loading study '{study_name}'...")
    
    db_url = os.getenv("HPO_DATABASE_URL", "sqlite:///hpo_studies.db")
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
    except KeyError:
        print(f"Error: Study '{study_name}' not found.")
        return

    # Load Pathfinder metadata
    pathfinder_results = {}
    with SessionLocal() as session:
        results = session.query(TrialResult).filter_by(study_name=study_name).all()
        for r in results:
            pathfinder_results[r.trial_id] = r.to_dict()

    # Filter and flatten completed trials with resolution >= 500
    rows = []
    headers = [
        "trial_number", "trial_id", "state", "score", "loss", 
        "duration_seconds", "epoch_reached", "learning_rate", 
        "batch_size", "resolution", "encoder_name", "loss_weight_ratio", 
        "model_capacity", "gpu_model", "max_vram_gb"
    ]
    
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
            
        params = t.params
        res = params.get("resolution")
        if res is None or res < 500:
            continue
            
        trial_id = t._trial_id
        pf_info = pathfinder_results.get(trial_id, {})
        
        # Primary score and loss from Optuna values or Pathfinder
        values = t.values
        optuna_loss = values[0] if values and len(values) > 0 else None
        optuna_score = values[1] if values and len(values) > 1 else None
        
        score = pf_info.get("primary_score") or optuna_score
        loss = pf_info.get("primary_loss") or optuna_loss
        
        row = {
            "trial_number": t.number,
            "trial_id": trial_id,
            "state": t.state.name,
            "score": score,
            "loss": loss,
            "duration_seconds": t.duration.total_seconds() if t.duration else None,
            "epoch_reached": pf_info.get("epoch_reached"),
            "learning_rate": params.get("learning_rate"),
            "batch_size": params.get("batch_size"),
            "resolution": res,
            "encoder_name": params.get("encoder_name"),
            "loss_weight_ratio": params.get("loss_weight_ratio"),
            "model_capacity": params.get("model_capacity", "N/A"),
            "gpu_model": pf_info.get("gpu_model"),
            "max_vram_gb": pf_info.get("max_vram_gb")
        }
        rows.append(row)
        
    # Sort by trial number
    rows.sort(key=lambda x: x["trial_number"])
    
    # Save to CSV
    csv_file = "bridge_crack_study_500px.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    print(f"Successfully extracted {len(rows)} trials to '{csv_file}'.")
    
    if len(rows) > 0:
        first = rows[0]
        best = max(rows, key=lambda x: x["score"] if x["score"] is not None else 0)
        print("\n--- RESULTS OVERVIEW (Resolution >= 500px) ---")
        print(f"First Trial #{first['trial_number']}: Score={first['score']}, Loss={first['loss']}, Params={first['learning_rate'], first['batch_size'], first['resolution']}")
        print(f"Best Trial #{best['trial_number']}: Score={best['score']}, Loss={best['loss']}, Params={best['learning_rate'], best['batch_size'], best['resolution']}")
        if first['score'] is not None and best['score'] is not None:
            print(f"Improvement: {first['score']} -> {best['score']} (Gain: +{best['score'] - first['score']:.6f})")
    else:
        print("No completed trials matching criteria found.")

if __name__ == "__main__":
    extract_filtered_csv()
