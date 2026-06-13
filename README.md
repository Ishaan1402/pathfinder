# Pathfinder

[Python 3.10+](https://www.python.org/downloads/)
[FastAPI](https://fastapi.tiangolo.com/)
[Optuna](https://optuna.org/)
[SQLite](https://www.sqlite.org/)
[MCP](https://modelcontextprotocol.io/)

A decoupled hyperparameter optimization (HPO) framework that separates the deterministic optimizer from episodic AI reviews. Train workers run autonomously without ever blocking on an LLM. Optimizers run fast. Humans (or AI agents in your IDE) review results periodically and decide when to adjust the search space.

**Designed for:** ML researchers and engineers tuning deep learning models on their own infrastructure (local GPU, Colab, cloud VMs). Use it as a reference for the bridge-crack U-Net project, or adapt the templates for your own training script.

## Why Pathfinder?

**Problem:** Traditional HPO frameworks either require workers to wait for an optimizer, or they add LLM reasoning that introduces latency into every training loop. You end up trading off between speed and intelligence.

**Solution:** Three independent layers:

- **Broker (Optuna TPE)**: Fast, deterministic suggestion engine. Never calls an LLM. Workers hit this endpoint and move on.
- **Worker**: Train autonomously. Report metrics incrementally. Handles pruning, OOM, checkpointing. Never waits.
- **Coordinator (You + Optional LLM)**: Run episodic reviews when *you* decide. Inspect trial history, check search health, propose bounds changes. AI agents (Claude, Cursor) can run reviews via MCP tools.

All state lives in **SQLite**—no config files, no in-memory state. This makes it easy to resume reviews, audit decisions, and sync across machines.

## Quick Start

### Option 1: Local GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: Start broker
python broker.py --daemon
# Dashboard: http://127.0.0.1:8000

# Terminal 2: Run worker
HPO_BROKER_URL=http://localhost:8000 HPO_STUDY_NAME=test_study python simulators/training_worker.py
Option 2: Remote Workers (Colab / Cloud GPU)
```

### Option 2: Remote Workers (Colab / Cloud GPU)

```bash
# Terminal 1: Start broker with tunnel + auth
export HPO_SECRET_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# ngrok (auto-generates URL)
python broker.py --daemon --tunnel

# OR Cloudflare (bring your own domain)
python broker.py --daemon --tunnel-provider cloudflare --tunnel-url https://your-domain.com

# Prints: 🔥 Remote broker URL established: https://...

# Terminal 2 (on remote): Set environment and run worker
export HPO_BROKER_URL="https://..."
export HPO_SECRET_TOKEN="<your-token>"
python colab_worker.py
```

### 4. (Optional) Use IDE Integration

Point Claude Code, Cursor, or Antigravity to the MCP server for agent-driven onboarding and reviews. See **IDE Setup** below.

---

## Core Features

### Deterministic Optimizer (Hot Path)

- **TPE Sampler**: Probability-based hyperparameter suggestions (beats grid search)
- **ASHA Pruning**: Stop underperforming trials early to save GPU time
- **Multi-Objective Pareto**: Optimize for both Dice score *and* loss simultaneously
- **fANOVA Importances**: Which hyperparams actually matter? (→ guides your reviews)
- **No LLM calls**: Workers never block. Suggest latency is <10ms.

### Episodic Coordinator (You Decide When)

- Dashboard shows health warnings (nudges to review, never auto-reviews)
- 7-step review procedure: inspect data → rate health → adjust bounds if needed → submit audit trail
- Search space proposals staged for approval before taking effect
- Coordinator accuracy tracked: your reviews' forecasted score gains vs. measured deltas
- Optional LLM integration (Claude, Gemini, OpenAI) for automatic reviews

### State Machine (SQLite)

All configuration, trials, reviews, and metadata live in `hpo_studies.db`:

- Active search space (not on disk)
- Trial results + VRAM telemetry
- Coordinator review history with citations
- Study health tier and dismissal states
- Generated model cards

---

## For Your Own Project

If you cloned this to tune your model (not the bridge-crack reference):

1. **Write a manifest** (`train.hpo.yaml`):
  ```yaml
   study_name: my_study
   metrics:
     objectives:
       - name: loss
         direction: minimize
       - name: accuracy
         direction: maximize
   params:
     - name: learning_rate
       type: float_log
       min: 1e-5
       max: 1e-2
     - name: batch_size
       type: categorical
       options: [4, 8, 16, 32]
   worker:
     entrypoint: python train.py
  ```
2. **Register the study**:
  ```bash
   python hpo_cli.py validate train.hpo.yaml
   python hpo_cli.py init train.hpo.yaml
  ```
3. **Write a worker** (use `templates/worker_minimal.py` as template):
  ```python
   from src.hpo_client import TrialSession

   session = TrialSession(broker_url="http://localhost:8000", study_name="my_study")
   trial = session.suggest()

   for epoch in range(epochs):
       accuracy, loss = train_one_epoch(trial["params"])
       should_prune = session.report_epoch(epoch, score=accuracy, loss=loss)
       if should_prune:
           break

   session.complete(epoch, score=accuracy, loss=loss, state="COMPLETE")
  ```
4. **Run on your GPU** (set env vars first):
  ```bash
   export HPO_BROKER_URL=http://localhost:8000
   export HPO_STUDY_NAME=my_study
   python train.py
  ```

Full integration walkthrough: [docs/INTEGRATION.md](docs/INTEGRATION.md)

---

## IDE Setup (Agent-Driven Onboarding & Reviews)

### Cursor

1. **Settings → Features → MCP**
2. **+ Add New MCP Server**
3. Name: `pathfinder`
  Type: `command`  
   Command: `source .venv/bin/activate && python3 hpo_mcp_server.py`

### Claude Code / Antigravity

Add to your MCP config (`~/.config/claudecode/mcp_config.json` or similar):

```json
{
  "mcpServers": {
    "pathfinder": {
      "command": "python3",
      "args": ["hpo_mcp_server.py"],
      "env": {
        "HPO_DATABASE_URL": "sqlite:///./hpo_studies.db"
      }
    }
  }
}
```

Then tell any agent:

- **"integrate HPO"** → agent drafts manifest, validates, registers study
- **"run a coordinator review"** → agent fetches study data, rates health, proposes bounds changes

See [AGENTS.md](AGENTS.md) for the full procedure.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  You + Optional IDE Agent               │
│  - Manual reviews or @pathfinder tools  │
│  - Dashboard inspection                 │
└─────────────────────────────────────────┘
              ↕ MCP + HTTP
┌─────────────────────────────────────────┐
│  Broker (broker.py on localhost:8000)   │
│  - Optuna TPE suggestion engine         │
│  - Trial lifecycle (/api/suggest,       │
│    /api/report_epoch, /api/complete)    │
│  - Dashboard serving                    │
└─────────────────────────────────────────┘
              ↕ SQLite
┌─────────────────────────────────────────┐
│  hpo_studies.db                         │
│  - All state (search space, trials,     │
│    reviews, health, config)             │
└─────────────────────────────────────────┘
              ↕ HTTP
┌─────────────────────────────────────────┐
│  Training Workers (Any Box)             │
│  - Colab, local GPU, cloud VM           │
│  - 3-call client API                    │
│  - Never blocks on optimizer or LLM     │
└─────────────────────────────────────────┘
```

---

## Common Commands

```bash
# Start broker + dashboard
python broker.py --daemon

# Validate & initialize a study from manifest
python hpo_cli.py validate train.hpo.yaml
python hpo_cli.py init train.hpo.yaml

# Check study health
python hpo_cli.py status

# Run a manual coordinator review (or prints prompt for copy-paste)
python hpo_cli.py review

# Export study config back to YAML
python hpo_cli.py manifest my_study

# Commit pending search space changes
python hpo_cli.py apply

# Run tests
pytest tests/ -q
```

---

## Reference: Bridge Crack Segmentation (bridge-crack repo)

This Pathfinder instance was initially tuned for [crack-seg](https://github.com/Ishaan1402/crack-seg#crack-seg), a **U-Net pixel-level crack detection model** on high-res UAV bridge imagery. See [colab_worker.py](colab_worker.py) for the full reference implementation (dataset download, model setup, training loop).

**Don't modify `colab_worker.py`** unless you're maintaining the bridge-crack project. Cloners should use `templates/worker_minimal.py` instead.

---

## Docs

- **[AGENTS.md](AGENTS.md)** — Guide for AI agents (Claude, Cursor, Antigravity)
- **[CLAUDE.md](CLAUDE.md)** — Development commands and architecture for Claude Code
- **[examples/onboarding/](examples/onboarding/)** — Step-by-step walkthrough for a new project
- **[docs/INTEGRATION.md](docs/INTEGRATION.md)** — Worker integration contract details

---

## License

MIT License - see the [LICENSE](LICENSE) file for details.