# AGENTS.md - Pathfinder

Canonical, IDE-agnostic guide for AI agents working in this repo. It applies the same in
**Cursor, Antigravity, and Claude Code** once the MCP server is configured (see "MCP setup"
below). There is one procedure, not three.

## Product split

This project is **not** "an LLM that picks hyperparameters." It is three layers:

| Layer | Responsibility | LLM involved? |
|-------|----------------|---------------|
| Broker + Optuna (`broker.py`) | Fast, reproducible HPO: suggest (TPE), prune, Pareto, fANOVA, fixed-eval honesty. The hot path. | No |
| GPU/Training worker (local training script, remote server, or Colab) | Execute trials, report metrics. Never waits on an LLM. | No |
| Episodic coordinator (this guide + MCP tools) | Interpret results, flag drift, apply bounded policy changes, scaffold onboarding. | Yes, on user request |

MLE correctness lives in the engine; AI value lives in interpretation, policy under
guardrails, and onboarding. **Never block the worker on a model call. Never auto-run the
coordinator without an explicit user request.**

## MCP setup (one time, shared across IDEs)

All three IDEs use the same server `pathfinder` (`python hpo_mcp_server.py`) and the
same database. See the README "IDE Setup (Agent-Driven Onboarding & Inspection)" section for the exact config block.
Capability does not differ between Cursor, Antigravity, and Claude Code. Use one IDE at a time
for writes; reviews are idempotent per trial window, so a second client will not double-write.

## Onboarding procedure

Trigger this when the user says "integrate HPO", "onboard my training script", "wire hyperparameter tuning", or clones the repo fresh. **Offer a one-line "Run Pathfinder onboarding?" and do not write files until the user confirms.**

1. **Read the user's training entrypoint.** Identify tunable hyperparameters, metric names, and metric directions.

2. **Generate a manifest file.** Write `train.hpo.yaml` (or the user's chosen name) using the schema in `templates/manifest.template.yaml`. Do NOT call registration tools (`init_from_manifest`) yet. Show the user the file for review first, though you may call `validate_manifest` to verify your draft is syntactically valid.

3. **User reviews the manifest.** They check bounds, metric names, and directions. They edit anything that looks wrong. The manifest is theirs, not yours.

4. **Validate mechanically.** If you are running as an MCP-enabled IDE agent (Cursor, Antigravity, Claude Code), call the MCP tool `validate_manifest(yaml_str)` to check for errors/warnings. Alternatively, tell the user to run:
   ```
   python hpo_cli.py validate train.hpo.yaml
   ```
   If validation fails, read the errors, fix the manifest, repeat.

5. **Register the study.** Call the MCP tool `init_from_manifest(yaml_str, force=...)` to register the study in SQLite and Optuna. Alternatively, tell the user to run:
   ```
   python hpo_cli.py init train.hpo.yaml
   ```
   This creates the Optuna study and stores config in the database.

6. **Generate the worker.** From `templates/worker_minimal.py`, write a worker script that uses `TrialSession`. Note that the worker reports metrics using the generic `report_epoch(epoch, score=..., loss=...)` and `complete(..., score=..., loss=...)` slots. Do NOT pass custom objective names as arguments; instead, map your higher-is-better metric to `score` and your lower-is-better metric to `loss`.

7. **Document the GPU side.** Tell the user to set `HPO_BROKER_URL` and `HPO_STUDY_NAME` on the training machine. The study name must match the manifest. Enable `HPO_SPARKLINES=1` if they want a Unicode performance curve printed on trial completion.

The worker contract is exactly three calls; full reference in `docs/INTEGRATION.md`.

### Statistical confidence (caveat, not a gate)

`get_study_data` includes `statistical_confidence` derived from completed-trial count:

| Tier | Completed trials | Agent behavior |
|------|------------------|----------------|
| `low` | &lt; 10 | Treat fANOVA/Spearman as noisy; be cautious with interpretation |
| `medium` | 10–19 | Signals stabilizing; stay cautious on large bound shifts |
| `high` | ≥ 20 | Standard interpretation |

Reviews are never hard-blocked at low confidence — the dashboard shows a banner only.

## Inspection flow

When the user asks about study progress, trial results, or health:

1. Call `get_study_data(study_name)` to retrieve trial telemetry, health tier, fANOVA importances, best trials, and OOM patterns.
2. Summarize: current best score, health status, stagnation indicators, and any OOM warnings.
3. If health is `watch` or `intervene`, explain the reason and offer to explore the data further.

## IDE triggers & status polling (.hpo_status.json)

When a completed trial is reported or the background daemon polls health, the system writes `.hpo_status.json` to the workspace root.

**Agent Action (Cursor, Antigravity, and Claude Code)**:
* Eagerly check for the existence of `.hpo_status.json` in the project root.
* If `health_tier` is `"watch"` or `"intervene"`, immediately notify the user in your first response, explain the `health_reason`, and proactively offer to inspect the study data.
* Offer to use the `/goal` slash command to help coordinate and automate resolution of search space stagnation or failures.

## Guardrails (always)

- Never block the GPU worker on an LLM; the suggest path stays TPE.
- Never auto-invoke the coordinator from the broker, dashboard, or a hook - only on user request.
- Do not write/modify local JSON files for search space or configuration (e.g. `active_search_space.json`, `hpo_config.json`); state must reside in SQLite.
- Do not modify the root `colab_worker.py` unless the user owns the bridge-crack project. Cloners use `templates/`.
- Never modify `archive/` files — they are historical reference only.
