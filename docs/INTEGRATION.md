# Integrating your training script with Pathfinder

A human quickstart for wiring your own model to the broker. For the agent-driven version of
this (have your IDE assistant do it), see [`AGENTS.md`](../AGENTS.md).

The reference crack-seg implementation is not included in this repository.
You do **not** need to fork it. Cloners start from `templates/worker_minimal.py` and the 3-call
client in `src/hpo_client.py`.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the broker

**Local only** (worker on the same machine — Colab cannot reach this):

```bash
source .venv/bin/activate
python broker.py --daemon   # http://127.0.0.1:8000, no token required
```

**Remote GPU (Colab, cloud VM)** — the broker must be internet-reachable, so auth is mandatory.
Refusing to start without a token is expected, not a bug:

```bash
export HPO_SECRET_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python broker.py --daemon --tunnel
```

Save that token and use the **same** value in three places:

| Where | How |
|-------|-----|
| Colab / worker | `os.environ["HPO_SECRET_TOKEN"] = "…"` — `TrialSession` sends `X-HPO-Token` |
| Dashboard | First visit to the tunnel URL prompts once; stored in a session cookie |
| CLI / MCP | `export HPO_SECRET_TOKEN=…` when tools hit the tunneled broker |

Worker downloads also require the token header when auth is on. The dashboard **Worker Setup** tab
generates copy-paste snippets.

## 3. Define your search space via a Manifest

Pathfinder uses a manifest-based onboarding system. You define your study config, search space
parameters, objectives, and training command in a single YAML file (e.g. `train.hpo.yaml`).

To set up a study:

1. **Create the manifest file**: Write a YAML file based on `templates/manifest.template.yaml`.
2. **Validate the manifest**:
   - **CLI**: `python hpo_cli.py validate train.hpo.yaml`
   - **MCP**: Use the `validate_manifest` tool.
   - **Dashboard**: Drag and drop the YAML into the **New Study** modal.
3. **Register/Initialize the study**:
   - **CLI**: `python hpo_cli.py init train.hpo.yaml`
   - **MCP**: Use the `init_from_manifest` tool.
   - **Dashboard**: Click **Initialize Study** after validating.

This stores configuration in the database and creates the Optuna study. The database
configuration keeps all studies isolated.

## 4. The worker contract (three calls)

Set environment variables on the machine that trains:

```bash
export HPO_BROKER_URL="https://<your-tunnel>"   # or http://localhost:8000
export HPO_STUDY_NAME="my_study"
export HPO_SPARKLINES=1                         # Optional: prints a Unicode curve on completion
```

Then use `src.hpo_client.TrialSession`:

```python
from src.hpo_client import TrialSession
import sys

# 1. Detect GPU telemetry
gpu_model = "CPU"
max_vram_gb = 0.0
try:
    import torch
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        max_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
except ImportError:
    pass

session = TrialSession()                 # reads HPO_BROKER_URL / HPO_STUDY_NAME
trial = session.suggest()                # -> {trial_id, trial_number, params}

pruned = False
oom_triggered = False
last_epoch = 0
score, loss = 0.0, 0.0

try:
    for epoch in range(num_epochs):
        last_epoch = epoch
        score, loss = train_one_epoch(trial["params"])    # your training step
        if session.report_epoch(epoch, score=score, loss=loss):  # True => broker says prune
            session.complete(
                epoch, score=score, loss=loss, state="PRUNED",
                gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
            )
            pruned = True
            break
except Exception as exc:
    # 2. Catch and report Out Of Memory (OOM) failures
    exc_str = str(exc).lower()
    if "out of memory" in exc_str or "oom" in exc_str:
        session.complete(
            last_epoch, score=score, loss=loss, state="FAIL",
            gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=True
        )
        print("Trial failed due to GPU OOM.")
        sys.exit(1)
    else:
        raise exc

if not pruned:
    session.complete(
        last_epoch, score=score, loss=loss, weights_path="model.pt", state="COMPLETE",
        gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
    )
```

That is the entire contract:

| Call | Endpoint | Purpose |
|------|----------|---------|
| `session.suggest()` | `POST /api/suggest_trial` | Get the next trial's hyperparameters |
| `session.report_epoch(epoch, score=score, loss=loss, ...)` | `POST /api/report_epoch` | Log an epoch; returns `should_prune` |
| `session.complete(epoch, score=score, loss=loss, ..., gpu_model, max_vram_gb, oom_triggered)` | `POST /api/complete_trial` | Finalize (COMPLETE / PRUNED / FAIL) with hardware telemetry |

`templates/worker_minimal.py` is a ~60-line starting point — fill in `train_one_epoch`.

> [!NOTE]
> **Abstracting Metrics (Non-CV Tasks)**
> The parameter names `score` (higher-is-better) and `loss` (lower-is-better) are generic slots in the API. If you are doing NLP (e.g. Perplexity and BLEU), RL (e.g. Reward and Episode Length), or Tabular tasks:
> - Pass your higher-is-better metric (e.g. Accuracy, BLEU, F1, Reward) as `score`.
> - Pass your lower-is-better metric (e.g. Cross-Entropy, Perplexity, MAE) as `loss`.
> - You can customize their display names on the UI dashboard under **Settings > Eval protocol** by setting "Loss metric display name" and "Score metric display name".

## 5. Create the study and validate

Call the MCP `init_from_manifest` tool (or CLI `init`) to create the Optuna study and seed
configuration options from your manifest file. Use the `/health` broker endpoint to verify
connectivity.

## 6. CLI operations

Pathfinder ships a command-line interface (`hpo_cli.py`) for database operations.

### Validate a manifest

```bash
python hpo_cli.py validate train.hpo.yaml
```

Parses and validates the YAML manifest against the Pathfinder schema. Reports errors and
warnings without touching the database.

### Initialize a study from manifest

```bash
python hpo_cli.py init train.hpo.yaml
python hpo_cli.py init train.hpo.yaml --force   # overwrite existing study
```

### Export a study

```bash
python hpo_cli.py export my_study --output my_study.json
python hpo_cli.py export my_study --format csv --output my_study.csv
```

Exports the full study — Optuna trials, trial results, reviews, agent logs — into a portable
JSON or CSV file. Useful for archiving, sharing, or migrating between machines.

### Import a study

```bash
python hpo_cli.py import my_study.json
python hpo_cli.py import my_study.json --rename new_study_name
python hpo_cli.py import my_study.json --rename new_study_name --force  # overwrite if exists
```

Imports a study from a previously exported JSON file. Any trials that were `RUNNING` at export
time are automatically converted to `FAIL` so they do not appear as zombie trials. If the import
fails partway through, the entire import is rolled back atomically.

### Generate a model card

```bash
python hpo_cli.py modelcard my_study
```

Writes `MODEL_CARD.md` to disk with a synthesis of the study's results, best hyperparameters,
and importance rankings.

### Delete a study

```bash
python hpo_cli.py delete my_study
```

Permanently removes a study and all its data from the database. Requires confirmation.

### Backup the database

```bash
python hpo_cli.py backup --output backup.db
```

Creates a point-in-time snapshot of the full SQLite database using SQLite's online backup API.
Safe to run while the broker is running.

## 7. Environment variables

| Variable | Default | Description |
|---|---|---|---|
| `HPO_DATABASE_URL` | `sqlite:///hpo_studies.db` | SQLite connection string. |
| `HPO_BROKER_URL` | `http://localhost:8000` | URL the worker uses to reach the broker. |
| `HPO_STUDY_NAME` | *(none)* | Default study name when not passed explicitly. |
| `HPO_SECRET_TOKEN` | *(none)* | Bearer token required when `--tunnel` auth is enabled. |
| `HPO_DEBUG` | `0` | Set to `1` to enable verbose debug logging in the broker. |
| `HPO_SPARKLINES` | `0` | Set to `1` to print a Unicode training curve on trial completion. |
| `HPO_BACKUP_ON_START` | `0` | Set to `1` to run an automatic database backup when the broker starts. |
| `HPO_CAPTURE_FULL_ENV` | `0` | Set to `1` to capture the full `pip freeze` output rather than the default ML-library whitelist. |
| `HPO_TUNNEL_PROVIDER` | *(none)* | Tunnel provider for remote access: `ngrok` or `cloudflare`. |
| `HPO_TUNNEL_URL` | *(none)* | Static tunnel URL when using `cloudflare` provider. |
| `HPO_ALLOWED_ORIGINS` | *(none)* | Additional CORS origins (comma-separated) for the dashboard. |

## 8. Validation guardrails schema

`validation_rules` can be set in the manifest YAML or via **Settings > Eval protocol >
Metric guardrails** in the dashboard.

```yaml
validation_rules:
  enabled: true          # master toggle — set false to disable all checks
  score_min: 0.0         # warn when score (higher-is-better objective) falls below this value
  loss_min: 0.0          # warn when loss (lower-is-better objective) falls below this value
  max_epoch_jump: 0.5    # warn when score changes by more than this fraction between consecutive epochs
```

When `enabled: false` (the default for new studies), no metric warnings are ever generated.
Set `enabled: true` only when you have domain knowledge about valid metric ranges for your task.

When a trial triggers a guardrail it is flagged as **Watch** in the study health, but the
trial is still recorded — guardrails are advisory, not blocking (the only hard rejection is
when *both* metrics are exactly `0.0`, history is empty, and `epoch ≤ 0` on a multi-objective
study, which strongly indicates training never ran).

## Next steps

- Open the dashboard (`index.html` served via the broker root) to watch trials, the Pareto front, and fANOVA importance.
- Use the inspection flow (see [`AGENTS.md`](../AGENTS.md)) to interpret results with your IDE agent.
