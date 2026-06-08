---
name: hpo-onboard
description: Onboard a training pipeline to Pathfinder. Use when the user wants to integrate HPO, onboard a training pipeline, set up a fresh clone, or wire hyperparameter tuning into their training script.
---

# Pathfinder Onboarding

Follow AGENTS.md onboarding section; use @pathfinder MCP tools; do not modify root colab_worker.py unless user owns bridge-crack project.

## Steps

1. Read [AGENTS.md](../../../AGENTS.md) "Onboarding procedure" and follow it.
2. Load the MCP resource `hpo://prompts/grill` for the canonical onboarding checklist.
3. Offer a one-line "Run Pathfinder onboarding?" and do not write files until the user confirms.
4. Scaffold from `templates/` (search space, hpo config, `worker_minimal.py`) using `hpo_client.TrialSession` - never fork `colab_worker.py`.
5. Finish with the MCP tool `validate_integration(study_name)`.
