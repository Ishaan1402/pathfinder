#!/usr/bin/env python3
"""Decoupled Control CLI for Pathfinder.

Provides standalone commands to check status, manage pending search space patches,
export/import studies, generate model cards, and delete studies.
"""
import os
import sys
import json
import argparse
import csv
import sqlite3
import datetime
from datetime import timezone
import optuna

# Make sure we can import from workspace root and src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.db_manager import get_db_session, init_db, DATABASE_URL
from src.schema import (
    StudyStatus,
    SystemConfiguration,
    TrialResult,
    TrialMetadata,
    CompactedPacket,
    StudyCard,
    TrialLease,
)
from src.health import compute_health_tier, count_evaluated_trials
from src.analytics import build_study_packet
from src.suggest import load_study

DEFAULT_STUDY = None

def get_study_name(args) -> str:
    """Resolve study name from args, env, or default."""
    name = args.study or os.getenv("HPO_STUDY_NAME")
    if not name:
        print("No study specified. Set HPO_STUDY_NAME or pass --study.")
        sys.exit(1)
    return name

def cmd_status(args):
    init_db()
    study_name = get_study_name(args)
    try:
        study = load_study(study_name)
    except Exception as e:
        print(f"Error loading study '{study_name}': {e}")
        sys.exit(1)

    health_tier, health_reason = compute_health_tier(study, study_name)

    print("\n==================================================")
    print(f"📊 STUDY STATUS: {study_name}")
    print("==================================================")
    print(f"Total Trials:      {len(study.trials)}")
    print(f"Evaluated:         {count_evaluated_trials(study)}")
    print(f"Health Tier:       {health_tier.upper()}")
    print(f"Health Reason:     {health_reason}")
    print("==================================================\n")


def cmd_validate(args):
    import yaml
    from src.manifest import validate_manifest
    path = args.manifest
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"✗ Invalid YAML: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"✗ File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading manifest: {e}")
        sys.exit(1)
    
    errors, warnings = validate_manifest(data)
    
    if warnings:
        for w in warnings:
            print(f"⚠  {w}")
    
    if errors:
        for e in errors:
            print(f"✗ {e}")
        print(f"\n{len(errors)} error(s) found. Fix them and re-run validate.")
        sys.exit(1)
    
    print(f"✓ Manifest is valid: {len(data.get('params', []))} parameters, "
          f"{len(data.get('metrics', {}).get('objectives', []))} objective(s)")
    sys.exit(0)

def cmd_quickstart(args):
    print("Welcome to Pathfinder Quickstart!")
    print("Let's get a dead-simple dummy optimization study running.\n")
    
    study_name_base = input("1. Study Name [quickstart_study]: ").strip() or "quickstart_study"
    param_name = input("2. Tunable Hyperparameter Name [x]: ").strip() or "x"
    param_min_str = input(f"   Minimum bound for {param_name} [-10.0]: ").strip() or "-10.0"
    param_max_str = input(f"   Maximum bound for {param_name} [10.0]: ").strip() or "10.0"
    direction = input("3. Optimization Direction (minimize/maximize) [minimize]: ").strip().lower() or "minimize"
    
    if direction not in ("minimize", "maximize"):
        print("✗ Invalid direction. Must be minimize or maximize.")
        sys.exit(1)
        
    try:
        p_min = float(param_min_str)
        p_max = float(param_max_str)
    except ValueError:
        print("✗ Bounds must be numerical.")
        sys.exit(1)
        
    init_db()
    with get_db_session() as session:
        from src.schema import StudyStatus
        # Auto-suffix handling
        study_name = study_name_base
        suffix = 2
        while session.query(StudyStatus).filter_by(study_name=study_name).first() is not None:
            if suffix > 5:
                print(f"\n✗ Error: Studies from {study_name_base} to {study_name_base}_5 already exist.")
                print("Please provide a different study name or delete an existing one.")
                sys.exit(1)
            study_name = f"{study_name_base}_{suffix}"
            suffix += 1

    if study_name != study_name_base:
        print(f"\nℹ Note: '{study_name_base}' existed, using '{study_name}' instead.")
        
    yaml_content = f"""study_name: {study_name}

metrics:
  primary_score: {('loss' if direction == 'minimize' else 'score')}
  objectives:
    - name: {('loss' if direction == 'minimize' else 'score')}
      direction: {direction}
      label: "Metric"

params:
  - name: {param_name}
    type: float
    min: {p_min}
    max: {p_max}

worker:
  entrypoint: python quickstart_worker.py
"""
    yaml_path = "quickstart.hpo.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    metric_name = "loss" if direction == "minimize" else "score"
    broker_url = os.getenv("HPO_BROKER_URL", "http://localhost:8000")
    worker_content = f"""import sys
from src.hpo_client import TrialSession

def main():
    session = TrialSession(broker_url="{broker_url}", study_name="{study_name}")
    trial = session.suggest()
    {param_name} = trial["params"]["{param_name}"]
    
    # Dummy math evaluation
    val = ({param_name} - 2) ** 2
    
    session.complete(epoch=0, {metric_name}=val)
    print(f"Trial {{trial.get('trial_number')}} complete: {param_name}={{val:.4f}} -> {metric_name}={{val:.4f}}")

if __name__ == "__main__":
    main()
"""
    worker_path = "quickstart_worker.py"
    with open(worker_path, "w") as f:
        f.write(worker_content)
        
    print(f"\nGenerated {yaml_path} and {worker_path}.")
    
    import yaml
    from src.onboarding import init_study_from_manifest_dict
    from src.manifest import validate_manifest
    
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        
    errors, warnings = validate_manifest(data)
    if errors:
        print("\n✗ Unexpected validation error in generated manifest:")
        for e in errors:
            print(f"✗ {e}")
        sys.exit(1)
        
    result = init_study_from_manifest_dict(data, force=False)
    print(result)
    
    print("\n==================================================")
    print("🚀 SUCCESS! Your dummy study is registered.")
    print("Run the following command in another terminal:")
    print("\n    python quickstart_worker.py")
    print("==================================================\n")
    sys.exit(0)

def cmd_init(args):
    import yaml
    from src.onboarding import init_study_from_manifest_dict
    from src.manifest import validate_manifest

    path = args.manifest
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Error reading manifest: {e}")
        sys.exit(1)

    errors, warnings = validate_manifest(data)
    if errors:
        for e in errors:
            print(f"✗ {e}")
        print(f"\n{len(errors)} error(s) found during validation. Fix them and re-try init.")
        sys.exit(1)

    init_db()

    if args.force:
        study_name = data.get("study_name")
        study_exists = False
        try:
            import optuna
            from src.db_manager import DATABASE_URL
            optuna.load_study(study_name=study_name, storage=DATABASE_URL)
            study_exists = True
        except KeyError:
            pass
        if study_exists:
            ans = input(f"Study '{study_name}' already exists. Are you sure you want to delete it? [y/N]: ").strip().lower()
            if ans != 'y':
                print("Aborted.")
                sys.exit(0)

    try:
        result = init_study_from_manifest_dict(data, force=args.force)
        print(result)
        
        worker_data = data.get("worker", {})
        if worker_data.get("entrypoint"):
            print(f"\nNext: set HPO_BROKER_URL and run: {worker_data['entrypoint']}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)

def cmd_manifest(args):
    from src.manifest import export_manifest_yaml
    
    study_name = args.study
    init_db()
    
    try:
        yaml_str = export_manifest_yaml(study_name)
        print(yaml_str)
        sys.exit(0)
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)

def cmd_export(args):
    init_db()
    study_name = get_study_name(args)
    fmt = args.format.lower()
    
    try:
        study = load_study(study_name)
    except Exception as e:
        print(f"✗ Error loading study '{study_name}': {e}")
        sys.exit(1)
        
    if fmt == "csv":
        if not args.output:
            print("✗ Error: --output file path is required for csv format export.")
            sys.exit(1)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        try:
            trials = study.trials
            with get_db_session() as session:
                results = {r.trial_id: r for r in session.query(TrialResult).filter_by(study_name=study_name).all()}
            
            param_names = sorted(list(set(k for t in trials for k in t.params.keys())))
            headers = ["trial_id", "trial_number", "state", "value", "values"] + param_names + [
                "epoch_reached", "primary_score", "primary_loss", "gpu_model", 
                "max_vram_gb", "oom_triggered", "worker_id", "git_commit", 
                "dataset_version", "health_tier", "health_reason"
            ]
            
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for t in trials:
                    res = results.get(t._trial_id)
                    row = [
                        t._trial_id,
                        t.number,
                        t.state.name,
                        t.value,
                        json.dumps(t.values) if t.values else None,
                    ]
                    for p in param_names:
                        row.append(t.params.get(p))
                    if res:
                        row += [
                            res.epoch_reached,
                            res.primary_score,
                            res.primary_loss,
                            res.gpu_model,
                            res.max_vram_gb,
                            res.oom_triggered,
                            res.worker_id,
                            res.git_commit,
                            res.dataset_version,
                            res.health_tier,
                            res.health_reason
                        ]
                    else:
                        row += [None] * 11
                    writer.writerow(row)
            print(f"✓ Successfully exported study '{study_name}' trials to '{args.output}' as CSV.")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Error exporting CSV: {e}")
            sys.exit(1)
            
    elif fmt == "json":
        export_data = {
            "study_name": study_name,
            "directions": [d.name for d in study.directions],
            "trials": [],
            "trial_results": [],
            "trial_metadata": [],
            "system_configuration": [],
            "compacted_packets": [],
            "study_cards": [],
            "study_status": [],
        }
        
        from optuna.distributions import distribution_to_json
        for t in study.trials:
            dists_serialized = {}
            for k, dist in t.distributions.items():
                dists_serialized[k] = distribution_to_json(dist)
                
            export_data["trials"].append({
                "trial_id": t._trial_id,
                "number": t.number,
                "state": t.state.name,
                "value": t.value if t.values is None or len(t.values) <= 1 else None,
                "values": t.values,
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
                "params": t.params,
                "distributions": dists_serialized,
                "user_attrs": t.user_attrs,
                "system_attrs": t.system_attrs,
                "intermediate_values": {str(k): v for k, v in t.intermediate_values.items()}
            })
            
        with get_db_session() as session:
            results = session.query(TrialResult).filter_by(study_name=study_name).all()
            export_data["trial_results"] = [r.to_dict() for r in results]
            
            metadata = session.query(TrialMetadata).filter_by(study_name=study_name).all()
            export_data["trial_metadata"] = [m.to_dict() for m in metadata]
            
            sys_configs = session.query(SystemConfiguration).filter_by(study_name=study_name).all()
            export_data["system_configuration"] = [sc.to_dict() for sc in sys_configs]
            
            packets = session.query(CompactedPacket).filter_by(study_name=study_name).all()
            export_data["compacted_packets"] = [
                {
                    "trials_evaluated": p.trials_evaluated,
                    "packet_json": p.packet_json,
                    "created_at": p.created_at.isoformat() if p.created_at else None
                }
                for p in packets
            ]
            
            cards = session.query(StudyCard).filter_by(study_name=study_name).all()
            export_data["study_cards"] = [c.to_dict() for c in cards]
            
            status = session.query(StudyStatus).filter_by(study_name=study_name).all()
            export_data["study_status"] = [s.to_dict() for s in status]
            
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"✓ Successfully exported study '{study_name}' JSON to '{args.output}'.")
        else:
            print(json.dumps(export_data, indent=2))
        sys.exit(0)

def cmd_import(args):
    init_db()
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"✗ Error: Import file '{file_path}' not found.")
        sys.exit(1)
        
    try:
        with open(file_path, "r") as f:
            export_data = json.load(f)
    except Exception as e:
        print(f"✗ Error reading import JSON: {e}")
        sys.exit(1)
        
    # JSON Validation
    if not isinstance(export_data, dict):
        print("✗ Error: Invalid import JSON format (root must be a dictionary).")
        sys.exit(1)
        
    original_study_name = export_data.get("study_name")
    new_study_name = args.rename or original_study_name
    
    if not original_study_name:
        print("✗ Error: Invalid import JSON format (missing study_name).")
        sys.exit(1)
        
    if not isinstance(export_data.get("trials"), list):
        print("✗ Error: Invalid import JSON format (trials must be a list).")
        sys.exit(1)
        
    if "directions" not in export_data:
        print("✗ Error: Invalid import JSON format (missing directions).")
        sys.exit(1)
        
    for t in export_data.get("trials", []):
        if not isinstance(t, dict) or "trial_id" not in t or "number" not in t or "state" not in t:
            print("✗ Error: Invalid import JSON format (trial objects must contain trial_id, number, and state).")
            sys.exit(1)
            
    try:
        optuna.load_study(study_name=new_study_name, storage=DATABASE_URL)
        if not args.force:
            print(f"✗ Error: Study '{new_study_name}' already exists. Use --force to overwrite.")
            sys.exit(1)
            
        ans = input(f"Study '{new_study_name}' already exists. Are you sure you want to delete it? [y/N]: ").strip().lower()
        if ans != 'y':
            print("Aborted.")
            sys.exit(0)
            
        print(f"Deleting existing study '{new_study_name}' as --force was specified...")
        optuna.delete_study(study_name=new_study_name, storage=DATABASE_URL)
        with get_db_session() as session:
            for model in [TrialResult, TrialMetadata, SystemConfiguration, CompactedPacket, StudyCard, StudyStatus, TrialLease]:
                session.query(model).filter_by(study_name=new_study_name).delete()
    except KeyError:
        pass
        
    directions = [d.lower() for d in export_data.get("directions", ["minimize", "maximize"])]
    try:
        study = optuna.create_study(
            study_name=new_study_name,
            storage=DATABASE_URL,
            directions=directions,
            load_if_exists=True
        )
    except Exception as e:
        print(f"✗ Error creating study '{new_study_name}': {e}")
        sys.exit(1)
        
    try:
        print("Importing Optuna trials...")
        trial_id_mapping = {}
        from optuna.trial import FrozenTrial, TrialState
        from optuna.distributions import json_to_distribution
        
        sorted_trials = sorted(export_data.get("trials", []), key=lambda t: t.get("number", 0))
        for t in sorted_trials:
            dt_start = datetime.datetime.fromisoformat(t["datetime_start"]) if t.get("datetime_start") else None
            dt_complete = datetime.datetime.fromisoformat(t["datetime_complete"]) if t.get("datetime_complete") else None
            
            dists = {}
            for param_name, dist_json_str in t.get("distributions", {}).items():
                dists[param_name] = json_to_distribution(dist_json_str)
                
            t_state_name = t["state"]
            if t_state_name == "RUNNING":
                t_state_name = "FAIL"
                if not dt_complete:
                    dt_complete = datetime.datetime.now(timezone.utc).replace(tzinfo=None)

            frozen_trial = FrozenTrial(
                number=t["number"],
                state=TrialState[t_state_name],
                value=t.get("value"),
                values=t.get("values"),
                datetime_start=dt_start,
                datetime_complete=dt_complete,
                params=t.get("params", {}),
                distributions=dists,
                user_attrs=t.get("user_attrs", {}),
                system_attrs=t.get("system_attrs", {}),
                intermediate_values={int(k): float(v) for k, v in t.get("intermediate_values", {}).items()},
                trial_id=t["trial_id"]
            )
            study.add_trial(frozen_trial)
            
            new_trial = study.trials[-1]
            trial_id_mapping[t["trial_id"]] = new_trial._trial_id
            
        print("Importing custom Pathfinder tables...")
        with get_db_session() as session:
            # Cache invalidation: delete old compacted packets
            session.query(CompactedPacket).filter_by(study_name=new_study_name).delete()
            
            for sc in export_data.get("system_configuration", []):
                session.add(SystemConfiguration(
                    study_name=new_study_name,
                    config_key=sc["config_key"],
                    config_value=sc["config_value"],
                    version=sc.get("version", 1)
                ))
                
            for r in export_data.get("trial_results", []):
                orig_trial_id = r["trial_id"]
                new_trial_id = trial_id_mapping.get(orig_trial_id)
                if new_trial_id is None:
                    continue
                
                created_at = datetime.datetime.fromisoformat(r["created_at"]) if r.get("created_at") else datetime.datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(TrialResult(
                    trial_id=new_trial_id,
                    study_name=new_study_name,
                    epoch_reached=r["epoch_reached"],
                    primary_score=r.get("primary_score"),
                    primary_loss=r.get("primary_loss"),
                    score_history_json=json.dumps(r.get("score_history", [])),
                    weights_path=r.get("weights_path"),
                    gpu_model=r.get("gpu_model"),
                    max_vram_gb=r.get("max_vram_gb"),
                    oom_triggered=r.get("oom_triggered"),
                    failure_tag=r.get("failure_tag"),
                    worker_id=r.get("worker_id"),
                    git_commit=r.get("git_commit"),
                    dataset_version=r.get("dataset_version"),
                    health_tier=r.get("health_tier"),
                    health_reason=r.get("health_reason"),
                    created_at=created_at
                ))
                
            for m in export_data.get("trial_metadata", []):
                orig_trial_id = m["trial_id"]
                new_trial_id = trial_id_mapping.get(orig_trial_id)
                if new_trial_id is None:
                    continue
                created_at = datetime.datetime.fromisoformat(m["created_at"]) if m.get("created_at") else datetime.datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(TrialMetadata(
                    trial_id=new_trial_id,
                    study_name=new_study_name,
                    meta_key=m["meta_key"],
                    meta_value=m["meta_value"],
                    created_at=created_at
                ))
                
            for c in export_data.get("study_cards", []):
                created_at = datetime.datetime.fromisoformat(c["created_at"]) if c.get("created_at") else datetime.datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(StudyCard(
                    study_name=new_study_name,
                    card_type=c["card_type"],
                    file_path=c["file_path"],
                    content_hash=c["content_hash"],
                    metadata_json=json.dumps(c.get("metadata", {})),
                    created_at=created_at
                ))
                
            for s in export_data.get("study_status", []):
                health_updated_at = datetime.datetime.fromisoformat(s["health_updated_at"]) if s.get("health_updated_at") else datetime.datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(StudyStatus(
                    study_name=new_study_name,
                    health_tier=s.get("health_tier", "healthy"),
                    health_reason=s.get("health_reason"),
                    health_updated_at=health_updated_at,
                    nudge_dismissed_trials=s.get("nudge_dismissed_trials")
                ))
    except Exception as err:
        print(f"✗ Error during import execution: {err}")
        print("Rolling back database transaction and deleting half-imported study...")
        try:
            from src.onboarding import delete_study_internal
            delete_study_internal(study_name=new_study_name, confirm=True)
        except Exception as delete_err:
            print(f"Failed to delete study: {delete_err}")
        sys.exit(1)
        
    print(f"✓ Successfully imported study '{new_study_name}' (mapped {len(trial_id_mapping)} trials).")
    sys.exit(0)

def cmd_backup(args):
    init_db()
    if args.output:
        dest_path = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = f"backups/hpo_backup_{timestamp}.db"
        
    db_path = DATABASE_URL.replace("sqlite:///", "") if DATABASE_URL.startswith("sqlite:///") else "hpo_studies.db"
    if not os.path.exists(db_path):
        print(f"✗ Error: Source database file '{db_path}' does not exist.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    try:
        src_conn = sqlite3.connect(db_path)
        dst_conn = sqlite3.connect(dest_path)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        print(f"✓ Backup created successfully: '{dest_path}'")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Backup failed: {e}")
        sys.exit(1)

def cmd_modelcard(args):
    import hashlib
    import os as _os

    study_name = args.study_name
    init_db()

    packet = build_study_packet(study_name)

    if not packet or packet.get("success") is False:
        print(f"Error: Could not build study packet for '{study_name}'")
        sys.exit(1)

    lines = []
    lines.append(f"# Model Card: {study_name}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.datetime.now(timezone.utc).replace(tzinfo=None).isoformat()}")
    lines.append("")

    lines.append("## Executive Summary")
    counts = packet.get("counts", {})
    lines.append(f"- Total Trials: {counts.get('total', 0)}")
    lines.append(f"- Completed: {counts.get('complete', 0)}")
    lines.append(f"- Pruned: {counts.get('pruned', 0)}")
    lines.append(f"- Failed: {counts.get('failed', 0)}")
    lines.append(f"- Statistical Confidence: {packet.get('statistical_confidence', 'unknown')}")
    lines.append("")

    lines.append("## Best Parameters")
    trial_bins = packet.get("trial_bins", {})
    elite = trial_bins.get("elite", [])
    if elite:
        best = elite[0]
        lines.append(f"- Best Trial: #{best.get('trial_id')}")
        lines.append(f"- Best Score: {best.get('primary_score')}")
        lines.append(f"- Best Loss: {best.get('primary_loss')}")
        lines.append("")
        lines.append("### Best Trial Parameters")
        for k, v in best.get("params", {}).items():
            lines.append(f"- `{k}`: {v}")
    lines.append("")

    fanova = packet.get("fanova_importances", {})
    if fanova:
        lines.append("## fANOVA Importances")
        for param, importance in sorted(fanova.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- `{param}`: {importance:.4f}")
        lines.append("")

    vram = packet.get("vram_telemetry", {})
    if vram:
        lines.append("## VRAM Telemetry")
        lines.append(f"- GPU Model: {vram.get('gpu_model', 'Unknown')}")
        lines.append(f"- GPU Capacity: {vram.get('gpu_capacity_gb', 'N/A')} GB")
        lines.append(f"- OOM Count: {vram.get('oom_count', 0)}")
        oom_risk = vram.get("bounds_oom_risk")
        if oom_risk:
            lines.append(f"- OOM Risk Level: {oom_risk.get('risk_level', 'N/A')}")
            lines.append(f"- Predicted Max VRAM: {oom_risk.get('predicted_max_vram_gb', 'N/A'):.2f} GB")
        lines.append("")

    health = packet.get("health", {})
    lines.append("## Health")
    lines.append(f"- Tier: {health.get('tier', 'unknown')}")
    lines.append(f"- Reason: {health.get('reason', 'N/A')}")
    lines.append("")

    content = "\n".join(lines)

    _os.makedirs("studies", exist_ok=True)
    file_path = f"studies/{study_name}_model_card.md"
    with open(file_path, "w") as f:
        f.write(content)

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    with get_db_session() as session:
        session.add(StudyCard(
            study_name=study_name,
            card_type="model_card",
            file_path=file_path,
            content_hash=content_hash,
            metadata_json=json.dumps({"generated_at": datetime.datetime.now(timezone.utc).replace(tzinfo=None).isoformat()})
        ))
        session.commit()

    print(f"Model card saved to {file_path} (hash: {content_hash[:12]}...)")

def cmd_delete(args):
    study_name = args.study_name
    if not args.confirm:
        print("Use --confirm to permanently delete the study")
        sys.exit(0)

    from src.onboarding import delete_study_internal
    result = delete_study_internal(study_name=study_name, confirm=True)
    print(result["message"])

def main():
    parser = argparse.ArgumentParser(description="Pathfinder CLI Control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    p_status = subparsers.add_parser("status", help="Get study health, trial counts, and pending status")
    p_status.add_argument("--study", help="Study name")

    # Validate
    p_validate = subparsers.add_parser("validate", help="Check manifest for errors")
    p_validate.add_argument("manifest", help="Path to manifest YAML file")

    # Quickstart
    subparsers.add_parser("quickstart", help="Interactive wizard to generate and initialize a dummy study")

    # Init
    p_init = subparsers.add_parser("init", help="Validate + register study")
    p_init.add_argument("manifest", help="Path to manifest YAML file")
    p_init.add_argument("--force", action="store_true", help="Force overwrite if study already exists")

    # Manifest
    p_manifest = subparsers.add_parser("manifest", help="Export study config to manifest YAML")
    p_manifest.add_argument("study", help="Study name")

    # Export
    p_export = subparsers.add_parser("export", help="Export HPO study trials and config")
    p_export.add_argument("--study", help="Study name")
    p_export.add_argument("--format", choices=["json", "csv"], default="json", help="Export format (default: json)")
    p_export.add_argument("--output", help="File path to save the export (required for csv)")

    # Import
    p_import = subparsers.add_parser("import", help="Import HPO study trials and config from JSON file")
    p_import.add_argument("file", help="Path to JSON file to import")
    p_import.add_argument("--rename", help="Rename the study during import")
    p_import.add_argument("--force", action="store_true", help="Overwrite study if it already exists")

    # Backup
    p_backup = subparsers.add_parser("backup", help="Create a safe online backup of the SQLite database")
    p_backup.add_argument("--output", help="Custom backup file path")

    # Modelcard
    p_modelcard = subparsers.add_parser("modelcard", help="Generate a model card for a study")
    p_modelcard.add_argument("study_name", help="Study name to generate model card for")

    # Delete
    p_delete = subparsers.add_parser("delete", help="Permanently delete a study and all its data")
    p_delete.add_argument("study_name", help="Study name to delete")
    p_delete.add_argument("--confirm", action="store_true", help="Confirm permanent deletion")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "quickstart":
        cmd_quickstart(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "backup":
        cmd_backup(args)
    elif args.command == "modelcard":
        cmd_modelcard(args)
    elif args.command == "delete":
        cmd_delete(args)

if __name__ == "__main__":
    main()
