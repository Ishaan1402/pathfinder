# Integrating your training script with Pathfinder

A human quickstart for wiring your own model to the broker. For the agent-driven version of
this (have your IDE assistant do it), see [`AGENTS.md`](../AGENTS.md).

The root `colab_worker.py` is the full bridge-crack U-Net reference implementation. You do
**not** need to fork it. Cloners start from `templates/worker_minimal.py` and the 3-call
client in `hpo_client.py`.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the broker

**Local only** (simulator on the same machine — Colab cannot reach this):

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
| Dashboard | First visit to the ngrok URL prompts once; stored in a session cookie |
| CLI / MCP | `export HPO_SECRET_TOKEN=…` when tools hit the tunneled broker |

Worker downloads (`/colab_worker.py`, `/hpo_client.py`) also require the token header when
auth is on. The dashboard **Worker Setup** tab explains this and generates copy-paste snippets.

## 3. Configure MCP (optional, for agent assistance)

Point your IDE's MCP config at `pathfinder` (`python hpo_mcp_server.py`). The same
config works in Cursor, Antigravity, and Claude Code - see the README
"Exposing Pathfinder to AI Agents" section. Set `HPO_BROKER_URL` in the MCP env if you want
`validate_integration` to check the live broker.

## 4. Define your search space via a Manifest

Pathfinder uses a manifest-based onboarding system. You define your study config, search space parameters, objectives, and training command in a single YAML file (e.g. `train.hpo.yaml`).

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

This stores configuration in the database and creates the Optuna study. The database configuration keeps all studies isolated.

## 5. The worker contract (three calls)

Set environment variables on the machine that trains:

```bash
export HPO_BROKER_URL="https://<your-tunnel>"   # or http://localhost:8000
export HPO_STUDY_NAME="my_study"
export HPO_SPARKLINES=1                         # Optional: prints a Unicode curve on completion
```

Then use `hpo_client.TrialSession`:

```python
from hpo_client import TrialSession
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
dice, bce = 0.0, 0.0

try:
    for epoch in range(num_epochs):
        last_epoch = epoch
        dice, bce = train_one_epoch(trial["params"])      # your training step
        if session.report_epoch(epoch, dice, bce):        # True => broker says prune
            session.complete(
                epoch, dice, bce, state="PRUNED",
                gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
            )
            pruned = True
            break
except Exception as exc:
    # 2. Catch and report Out Of Memory (OOM) failures
    if "out of memory" in str(exc).lower():
        session.complete(
            last_epoch, dice, bce, state="FAIL",
            gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=True
        )
        print("Trial failed due to GPU OOM.")
        sys.exit(1)
    else:
        raise exc

if not pruned:
    session.complete(
        last_epoch, dice, bce, weights_path="model.pt", state="COMPLETE",
        gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
    )
```

That is the entire contract:

| Call | Endpoint | Purpose |
|------|----------|---------|
| `session.suggest()` | `POST /api/suggest_trial` | Get the next trial's hyperparameters |
| `session.report_epoch(epoch, dice, bce, ...)` | `POST /api/report_epoch` | Log an epoch; returns `should_prune` |
| `session.complete(epoch, dice, bce, ..., gpu_model, max_vram_gb, oom_triggered)` | `POST /api/complete_trial` | Finalize (COMPLETE / PRUNED / FAIL) with hardware telemetry |

`templates/worker_minimal.py` is a ~65-line starting point - fill in `train_one_epoch`.

### Google Colab (bridge-crack reference)

The root `colab_worker.py` is **only** for the bridge-crack U-Net study. Cloners use
`templates/worker_minimal.py` above — do not fork `colab_worker.py`.

| Entrypoint | When to use |
|------------|-------------|
| `train_colab_trial(study_name, epochs=15)` | One trial (smoke test or debugging) |
| `train_colab_trial_loop(study_name, n_trials=12, epochs=15)` | Normal Colab session (default) |

`train_colab_trial_loop` repeatedly calls `train_colab_trial`. Each iteration:

- Reports guardrail skips and caught OOM/crashes as `FAIL` to the broker (loop continues).
- Clears the CUDA cache between trials.
- Retries transient `suggest_trial` errors (via `TrialSession` backoff).

If the Colab kernel hard-crashes without running Python cleanup, the broker marks the trial
`FAIL` after the worker lease expires (~45s without a heartbeat) or on the next dashboard poll.

```python
import os
os.environ["HPO_BROKER_URL"] = "https://<your-ngrok-url>"
os.environ["HPO_SECRET_TOKEN"] = "<same-token-as-broker>"  # required when --tunnel is on

from colab_worker import train_colab_trial_loop
train_colab_trial_loop("bridge_crack_study", n_trials=12, epochs=15)
```

The dashboard **Worker Setup → Google Colab Integration** tab generates the full
download-and-run snippet (including authenticated fetches of `colab_worker.py`).

> [!NOTE]
> **Abstracting Metrics (Non-CV Tasks)**
> The parameter names `dice` (higher-is-better) and `bce` (lower-is-better) are abstract placeholders in the API. If you are doing NLP (e.g. Perplexity and BLEU), RL (e.g. Reward and Episode Length), or Tabular tasks:
> - Pass your higher-is-better metric (e.g. Accuracy, BLEU, F1, Reward) as `dice`.
> - Pass your lower-is-better metric (e.g. Loss, Perplexity, MAE) as `bce`.
> - You can customize their display names on the UI dashboard under **Settings > Eval protocol** by setting "Loss metric display name" and "Score metric display name" (they default to BCE and Dice).

## 6. Create the study and validate

Call the MCP `init_from_manifest` tool (or CLI `init`) to create the Optuna study and seed configuration options from your manifest file. Use the `/health` broker endpoint to verify connectivity.

## Next steps

- Open the dashboard (`index.html` served via the broker root) to watch trials, the Pareto front, and fANOVA importance.
- Use the episodic coordinator review (see [`AGENTS.md`](../AGENTS.md)) to interpret results and adjust the search space with evidence.
