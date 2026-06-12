# Pathfinder

This project transforms a basic SQLite-backed Model Context Protocol (MCP) server into a state-of-the-art (SOTA) hyperparameter tuning and performance evaluation tool. It orchestrates trials for a **U-Net crack segmentation project** using a decentralized **Worker-Broker-Registry** pattern.

---

## 1. Architectural Strategy

The project implements a decoupled loop. Your IDE assistant or AI agent uses the MCP server to declare studies, query history, check AI performance metrics, and log reasoning. A decoupled training script (which runs anywhere: locally, remote GPU servers, Google Colab) queries the broker, executes the PyTorch training loop, and reports metrics back.

```
+---------------------------+         MCP         +---------------------------+
|    AI Assistant (IDE)     | <-----------------> |   FastMCP Server (Local)  |
+---------------------------+                     +---------------------------+
              |                                                 |
         Reads Stats                                      Read / Write
              v                                                 v
+-----------------------------------------------------------------------------+
|              Database (Local SQLite / Hosted Cloud PostgreSQL)              |
+-----------------------------------------------------------------------------+
                                      ^
                                 Read / Write
                                      |
                         +---------------------------+
                         |  Training Worker (Any Box)| ---> [ U-Net Pipeline ]
                         +---------------------------+
```

---

## 2. Advanced Search & Optimization Methods

- **Tree-structured Parzen Estimators (TPE)**: Used for conditional and continuous search space sampling to optimize BCE and Dice scores efficiently.
- **ASHA / Median Pruning**: Evaluates intermediate validation scores per epoch. If a trial falls below historical runs, it is immediately pruned to save compute.
- **Multi-Objective Optimization (Pareto Front)**: Simultaneously minimizes Binary Cross-Entropy (BCE) loss while maximizing Dice Score, displaying non-dominated configuration trade-offs.
- **fANOVA Parameter Importance**: Calculates the variance contribution of each hyperparameter to the segmentation metrics using functional ANOVA, enabling the AI to prioritize the right tuning parameters.
- **AI Decision Profiling**: Tracks AI convergence speed (trials to reach `Dice >= 0.85`), invalid parameter proposals (hallucination guardrails), and reasoning accuracy (correlation between estimated and actual dice improvement).

---

## 3. Database Schema

The SQLite database (`hpo_studies.db`) contains Optuna's internal trial logs alongside our custom context tables:

1. **`trial_results`**: Records training metrics per trial (epochs reached, score/loss, validation history, checkpoint paths, and GPU memory telemetry).
2. **`trial_metadata`**: Extensible key-value metadata storage associated with specific trials.
3. **`system_configuration`**: Stores key-value configuration states per study (active search space, HPO metrics config, project context, and source files).
4. **`agent_reasoning_logs`**: Logs the AI's predicted score improvements, rationales, and model/prompt version tags *before* training, and compares them to actual metrics *post-run* to calculate prediction MAE.
5. **`study_status`**: Tracks study-level health tiers (`healthy`, `watch`, `intervene`), reasons, and nudge dismissal state.
6. **`study_reviews`**: Stores one row per coordinator review (health rating, summary, policy actions, trigger reasons, baseline score, and actual score improvements).
7. **`compacted_packets`**: Caches compacted historical review data to optimize retrieval.
8. **`study_cards`**: Tracks the audit trail of synthesized report cards (`MODEL_CARD.md`).
9. **`invalid_proposals`**: Logs AI parameter validation failures (acting as hallucination guardrails).
10. **`trial_leases`**: Manages locks and leases for concurrent GPU training workers.
11. **`coordinator_metrics`** & **`suggest_metrics`**: Telemetry tables tracking server latency.

> [!NOTE]
> **Database-Backed Nudge Dismissal**
> To prevent desynced "Coordinator review suggested" notifications across different browsers or dev machines, the logo double-ping state is managed directly in the SQLite database through the `/api/dismiss_coordinator_nudge` endpoint. When a developer copies the review prompt or dismisses the notification banner on one machine, the status is updated in the SQLite database, immediately clearing the ping visual across all active dashboard sessions.

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10 or higher

### Install Dependencies
Set up the virtual environment and install all packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Exposing Pathfinder to AI Agents (MCP Settings)

### Claude Code & Antigravity 2.0
Add the server configurations to your global MCP settings file (typically `~/.config/claudecode/mcp_config.json` or custom Antigravity configuration):
```json
{
  "mcpServers": {
    "pathfinder": {
      "command": "python3",
      "args": [
        "hpo_mcp_server.py"
      ],
      "env": {
        "HPO_DATABASE_URL": "sqlite:///./hpo_studies.db"
      }
    }
  }
}
```

### Cursor Integration
1. Go to **Cursor Settings > Features > MCP**.
2. Click **+ Add New MCP Server**.
3. Fill in the following details:
   - **Name**: `pathfinder`
   - **Type**: `command`
   - **Command**: `source .venv/bin/activate && python3 hpo_mcp_server.py` (run from the repo root)
4. Now, any Cursor agent can access and execute your tuning tools using `@pathfinder`.

---

## 6. How to Run

### Run the HTTP Broker & Custom Dashboard
To start the FastAPI broker and serve the custom dashboard:
```bash
source .venv/bin/activate
python broker.py --daemon
```
Open `http://127.0.0.1:8000` in your web browser to access the custom dashboard.

### Run the Decoupled Worker
To run the local simulated worker:
```bash
source .venv/bin/activate
python simulators/training_worker.py
```

### Run on Google Colab (bridge-crack reference)
The root [`colab_worker.py`](colab_worker.py) is the full U-Net reference — not for cloners (use [`templates/worker_minimal.py`](templates/worker_minimal.py) instead). Two entrypoints:

| Function | Purpose |
|----------|---------|
| `train_colab_trial(study_name, epochs=15)` | One trial: suggest → train → complete / prune / fail |
| `train_colab_trial_loop(study_name, n_trials=12, epochs=15)` | Full session: calls `train_colab_trial` repeatedly |

Set `HPO_BROKER_URL` to your tunnel URL on Colab. The dashboard **Google Colab Integration** tab copies a ready-made snippet. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md#google-colab-bridge-crack-reference).

*Note: Set `HPO_BROKER_URL` on any remote GPU worker (Colab, cloud VM) so it reaches your broker.*

### Launch the Visualization Dashboard
To view interactive charts, Pareto-front curves, and hyperparameter importances:
```bash
source .venv/bin/activate
optuna-dashboard sqlite:///hpo_studies.db
```
Open the output URL (typically `http://127.0.0.1:8080`) in your browser.

### Run Integration Tests
To run the automated tests verifying TPE search, pruning, and fANOVA calculation:
```bash
source .venv/bin/activate
pytest tests/ -q          # full unit suite
python tests/test_integration.py   # end-to-end (real broker + worker)
```

---

## For cloners (use your own training script)

The root `colab_worker.py` is the **bridge-crack U-Net reference implementation**. The search
space, HPO config, and study state all live in **SQLite** (the `system_configuration` table) —
there are no live config files on disk. If you cloned this repo to tune your own model:

- Start from [`templates/worker_minimal.py`](templates/worker_minimal.py) and the 3-call
  client in [`hpo_client.py`](hpo_client.py) (`suggest` -> `report_epoch` -> `complete`).
- Define your search space, objectives, and training commands in a manifest file (e.g. `train.hpo.yaml`) using the template [`templates/manifest.template.yaml`](templates/manifest.template.yaml).
- Validate and initialize your study from the manifest YAML file. You can do this via the CLI (`python hpo_cli.py validate` and `python hpo_cli.py init`), drag-and-drop on the Dashboard, or via MCP.
- Human quickstart: [`docs/INTEGRATION.md`](docs/INTEGRATION.md).
- Agent-driven onboarding: [`AGENTS.md`](AGENTS.md). Tell any agent "integrate HPO" /
  "onboard my training script"; it uses `@pathfinder` (`validate_manifest` →
  `init_from_manifest` → `validate_integration`).

The same MCP config works in **Cursor, Antigravity, and Claude Code** - one setup, one
procedure (see "Exposing Pathfinder to AI Agents" above and `AGENTS.md`).

---

## 7. Coordinator Review (Agentic Layer)

The optimizer (Optuna TPE) stays the hot path: the Colab worker always asks the broker for the next trial and never waits on a language model. On top of that engine sits an **episodic coordinator** — an IDE agent (Cursor, Antigravity, or Claude Code, whichever you have open) that you invoke to interpret results, flag when the search policy looks wrong, and apply bounded changes. The same MCP server and the same SQLite database back all three IDEs, so capability does not change between them.

### What the layers do

| Layer | Responsibility |
|-------|----------------|
| Broker + Optuna | Fast, reproducible HPO: suggest, prune, Pareto, fANOVA, fixed-eval honesty. |
| GPU/Training worker | Execute trials on your local/remote GPU box (or Colab). Never blocks on an LLM. |
| Dashboard | Observability, and a deterministic "review suggested" nudge (no fake thinking). |
| MCP coordinator (one IDE at a time) | Rate search health, write an audit trail, and optionally narrow bounds or enqueue one trial. |

### When to invoke a review

You drive reviews; the machine only nudges. Run a review when any of these is true:

- The dashboard Analysis tab shows **Coordinator review suggested**.
- Roughly every five completed or pruned trials, as a habit.
- Before you change the search space or evaluation protocol in the UI.
- When you start a coding session that touches `broker.py`, `colab_worker.py`, or the study's search space / evaluation config.

The "review suggested" flag is computed deterministically in `hpo_coordinator.py` (train-vs-fixed-eval gaps, consecutive prunes, Dice stagnation, invalid-proposal spikes, or "enough trials but never reviewed"). No language model is called to raise it.

### The seven-step review procedure

Run this against `@pathfinder` in whichever IDE is open:

1. Call `get_study_data(study_name)` to retrieve the compacted review packet (which contains active search space, config, and study telemetry).
2. Read dynamic metric labels from the packet's `project_context`.
3. Rate search health 1–5, citing trials by number and preferring fixed-eval (deploy) Dice over train Dice.
4. Perform safety review of VRAM predictions and check coordinator accuracy.
5. Pick exactly one policy action: `no_change`, `update_search_space` (propose), or `enqueue_one_manual_trial`.
6. If proposing active search space changes, call `update_search_space(study_name, space_config, apply=False)`. If enqueuing a manual trial, pass the parameter dictionary as the `manual_trial` argument when calling `submit_agent_review`.
7. Call `submit_agent_review(study_name, summary, health_rating, policy_action, reasons=...)` to persist the audit trail.

### One IDE at a time, idempotent writes

If you happen to have both Cursor and Antigravity open, nothing double-fires: the broker never calls an LLM on its own, the dashboard nudge is a single web view, and `submit_agent_review` is **idempotent per trial window**. A second client submitting a review for the same number of finished trials receives the existing review (`duplicate: true`) instead of writing a new row, unless it passes `force=true`. Treat the coordinator as read-heavy and write-once per review cycle.


