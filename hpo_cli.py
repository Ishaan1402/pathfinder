#!/usr/bin/env python3
"""Decoupled Control CLI for Pathfinder.

Provides standalone commands to check status, run reviews, and manage pending search space patches.
"""
import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, Optional

# Make sure we can import from workspace root and src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.db_manager import get_db_session, init_db
from src.schema import StudyStatus, SystemConfiguration, StudyReview
from src.hpo_coordinator import (
    compute_review_heuristics,
    build_review_prompt,
    save_study_review,
    count_evaluated_trials,
    validate_review_fields,
    mark_review_applied,
    flag_study_review,
)
from src.hpo_config import load_hpo_config
from broker import get_or_create_study, load_search_space, _apply_search_space_patch, _enqueue_manual_trial

DEFAULT_STUDY = "bridge_crack_study"

def get_study_name(args) -> str:
    """Resolve study name from args, env, or default."""
    return args.study or os.getenv("HPO_STUDY_NAME") or DEFAULT_STUDY

def call_llm(prompt: str) -> str:
    """Call local LLM APIs directly using requests to avoid heavy client dependencies."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        res = requests.post(url, json=payload, headers=headers, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
        
    elif anthropic_key:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "system": "You are a professional ML experiment optimization coordinator. You MUST return JSON only."
        }
        res = requests.post(url, json=payload, headers=headers, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data["content"][0]["text"]
        
    elif openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a professional ML experiment optimization coordinator. You MUST return JSON only."},
                {"role": "user", "content": prompt}
            ]
        }
        res = requests.post(url, json=payload, headers=headers, timeout=120)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
        
    else:
        raise ValueError("No API keys found for GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")

def cmd_status(args):
    init_db()
    study_name = get_study_name(args)
    try:
        study = get_or_create_study(study_name)
    except Exception as e:
        print(f"Error loading study '{study_name}': {e}")
        sys.exit(1)

    from src.hpo_coordinator import study_eval_insights
    hpo_config = load_hpo_config(study_name)
    insights = study_eval_insights(study, hpo_config)
    heuristics = compute_review_heuristics(study, insights, hpo_config, study_name)

    print(f"\n==================================================")
    print(f"📊 STUDY STATUS: {study_name}")
    print(f"==================================================")
    print(f"Total Trials:      {len(study.trials)}")
    print(f"Evaluated:         {heuristics['trials_evaluated']}")
    print(f"Health Tier:       {heuristics['health_tier'].upper()}")
    print(f"Health Reason:     {heuristics['health_reason']}")
    print(f"Review Recommended: {heuristics['review_recommended']}")
    print(f"Already Dismissed: {heuristics.get('already_dismissed', False)}")
    
    # Check pending changes
    with get_db_session() as session:
        pending_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        if pending_row:
            print(f"Pending Changes:   YES (use 'python hpo_cli.py apply' to commit)")
        else:
            print(f"Pending Changes:   NO")
    print(f"==================================================\n")

def cmd_review(args):
    init_db()
    study_name = get_study_name(args)
    study = get_or_create_study(study_name)
    
    hpo_config = load_hpo_config(study_name)
    from src.hpo_coordinator import study_eval_insights
    insights = study_eval_insights(study, hpo_config)
    heuristics = compute_review_heuristics(study, insights, hpo_config, study_name)

    # Check if review already completed
    n_eval = heuristics["trials_evaluated"]
    if not args.force and heuristics["already_reviewed"]:
        print(f"Info: Study has already been reviewed for trial count {n_eval}. Use --force to override.")
        return

    # Check if API keys are set
    has_keys = any(os.getenv(k) for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    
    if not has_keys:
        # Just print the prompt
        print(f"No LLM API keys found (GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY).")
        print(f"Printing coordinator review prompt below for manual copy-paste:\n")
        print("----------------------------------------------------------------------")
        print(build_review_prompt(study_name))
        print("----------------------------------------------------------------------")
        return

    print("Running background LLM coordinator review...")
    prompt = build_review_prompt(study_name)
    
    try:
        response_text = call_llm(prompt)
        # Parse JSON output from LLM
        review_data = json.loads(response_text)
    except Exception as e:
        print(f"Failed to generate or parse LLM review: {e}")
        sys.exit(1)

    summary = review_data.get("summary", "LLM Generated review")
    health_rating = review_data.get("health_rating", 3)
    policy_action = review_data.get("policy_action", "no_change")
    reasons = review_data.get("reasons", [])
    est_imp = review_data.get("estimated_score_improvement") or review_data.get("estimated_dice_improvement")
    cited_best = review_data.get("cited_best_trial")
    patch = review_data.get("search_space_patch")
    manual_trial = review_data.get("manual_trial")

    validation = validate_review_fields(est_imp, cited_best)
    if not validation["ok"]:
        print(f"Review JSON contract error: {'; '.join(validation['errors'])}")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"🤖 LLM COORDINATOR REVIEW COMPLETED")
    print(f"==================================================")
    print(f"Health Rating: {health_rating}/5")
    print(f"Action:        {policy_action.upper()}")
    print(f"Summary:       {summary}")
    if patch:
        print(f"Space Patch:   {json.dumps(patch)}")
    if manual_trial:
        print(f"Manual Trial:  {json.dumps(manual_trial)}")
    print(f"==================================================")

    # Persist the review
    try:
        result = save_study_review(
            study_name,
            summary,
            health_rating=health_rating,
            policy_action=policy_action,
            model_version="cli_coordinator",
            reasons=reasons,
            trials_evaluated=n_eval,
            estimated_score_improvement=est_imp,
            cited_best_trial=cited_best,
            force=args.force
        )
        
        applied = {}
        space = load_search_space(study_name)
        
        # Save bounds proposal to pending config or apply it
        if patch:
            with get_db_session() as session:
                session.merge(SystemConfiguration(
                    study_name=study_name,
                    config_key="pending_search_space",
                    config_value=json.dumps(patch)
                ))
                session.commit()
            print("Proposed search space patch staged in 'pending_search_space'. Approve on dashboard or run 'python hpo_cli.py apply'.")

        if manual_trial:
            applied["manual_trial"] = _enqueue_manual_trial(study, manual_trial, space, summary)
            print(f"Enqueued manual trial: {manual_trial}")

        print("Review successfully saved in SQLite.")
    except Exception as e:
        print(f"Error persisting review: {e}")
        sys.exit(1)

def cmd_apply(args):
    init_db()
    study_name = get_study_name(args)
    
    with get_db_session() as session:
        pending_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        if not pending_row:
            print("No pending search space changes found.")
            return

        proposed = json.loads(pending_row.config_value)
        space = load_search_space(study_name)
        
        # Merge changes into active space
        for key, new_val in proposed.items():
            if key in space:
                p_type = space[key].get("type")
                if p_type == "categorical":
                    if "active" in new_val:
                        space[key]["active"] = new_val["active"]
                else:
                    if "min" in new_val:
                        space[key]["min"] = float(new_val["min"])
                    if "max" in new_val:
                        space[key]["max"] = float(new_val["max"])

        session.merge(SystemConfiguration(
            study_name=study_name,
            config_key="active_search_space",
            config_value=json.dumps(space)
        ))
        session.delete(pending_row)
        session.commit()

    mark_review_applied(study_name)
    print("Pending search space changes committed successfully.")


def cmd_flag_review(args):
    init_db()
    result = flag_study_review(args.id, flagged=not args.unflag)
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        sys.exit(1)
    state = "flagged" if not args.unflag else "unflagged"
    print(f"Review #{args.id} {state}.")

def cmd_discard(args):
    init_db()
    study_name = get_study_name(args)
    
    with get_db_session() as session:
        pending_row = session.query(SystemConfiguration).filter_by(
            study_name=study_name, config_key="pending_search_space"
        ).first()
        if not pending_row:
            print("No pending search space changes found.")
            return
        session.delete(pending_row)
        session.commit()
        
    print("Pending search space changes discarded.")

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

def cmd_init(args):
    import yaml
    from src.onboarding import init_study_from_manifest_dict

    path = args.manifest
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ Error reading manifest: {e}")
        sys.exit(1)

    init_db()

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
    from hpo_mcp_server import export_manifest
    
    study_name = args.study
    init_db()
    
    try:
        yaml_str = export_manifest(study_name)
        print(yaml_str)
        sys.exit(0)
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pathfinder CLI Control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    p_status = subparsers.add_parser("status", help="Get study health, trial counts, and pending status")
    p_status.add_argument("--study", help="Study name")

    # Review
    p_review = subparsers.add_parser("review", help="Execute coordinator review or output review prompt")
    p_review.add_argument("--study", help="Study name")
    p_review.add_argument("--force", action="store_true", help="Force review generation even if already completed for current trials")

    # Apply
    p_apply = subparsers.add_parser("apply", help="Commit pending search bounds configuration")
    p_apply.add_argument("--study", help="Study name")

    # Discard
    p_discard = subparsers.add_parser("discard", help="Discard pending search bounds configuration")
    p_discard.add_argument("--study", help="Study name")

    # Flag review quality
    p_flag = subparsers.add_parser("flag-review", help="Flag a coordinator review as low-quality (excluded from MAE)")
    p_flag.add_argument("--id", type=int, required=True, help="StudyReview row id")
    p_flag.add_argument("--unflag", action="store_true", help="Remove quality flag")

    # Validate
    p_validate = subparsers.add_parser("validate", help="Check manifest for errors")
    p_validate.add_argument("manifest", help="Path to manifest YAML file")

    # Init
    p_init = subparsers.add_parser("init", help="Validate + register study")
    p_init.add_argument("manifest", help="Path to manifest YAML file")
    p_init.add_argument("--force", action="store_true", help="Force overwrite if study already exists")

    # Manifest
    p_manifest = subparsers.add_parser("manifest", help="Export study config to manifest YAML")
    p_manifest.add_argument("study", help="Study name")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "apply":
        cmd_apply(args)
    elif args.command == "discard":
        cmd_discard(args)
    elif args.command == "flag-review":
        cmd_flag_review(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "manifest":
        cmd_manifest(args)

if __name__ == "__main__":
    main()
