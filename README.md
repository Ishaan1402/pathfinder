# Pathfinder

[![Build Status](https://github.com/Ishaan1402/pathfinder/actions/workflows/integration.yml/badge.svg)](https://github.com/Ishaan1402/pathfinder/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

Pathfinder is an MCP-integrated hyperparameter optimization dashboard that lets coding agents onboard your training script and inspect running experiments. It wraps Optuna's TPE sampler in a FastAPI broker with SQLite persistence and a dashboard UI. Study data is exposed through Model Context Protocol tools so your IDE agent can meaningfully participate and advise in the tuning loop.


<table border="0">
  <tr>
    <td width="67%" valign="top">
      <img src="docs/images/pathfinder_dashboard_example.png" alt="Pathfinder Dashboard" />
    </td>
    <td width="33%" valign="top">
      <img src="docs/images/pathways_plot.png" alt="Hyperparameter Pathways Plot" style="margin-bottom: 6px;" />
      <img src="docs/images/pruning_timeline.png" alt="Pruning Timeline" />
    </td>
  </tr>
</table>


## Why Pathfinder?

ML practitioners spend varying amounts of time and compute on poorly-bounded search spaces and have to manually inspect trial data by reading logs or refreshing notebooks. Pathfinder offers a live monitoring dashboard plus an MCP server so your IDE agent can read study state and help onboard new studies. 

2 layers:

- **Broker (Optuna TPE)**: Fast, deterministic suggestion engine. Suggestions and pruning happen in <10ms. Workers hit the broker and continue training immediately.
- **Worker**: Trains your model autonomously in a loop. Reports metrics per epoch, handles pruning, OOM detection, and checkpointing.

An MCP server lets coding agents inspect structured study data, validate manifests, and register new studies, only when you ask. The tuning path is never blocked by LLMs.

All state lives in SQLite.

## Quick Start



### Step 1: Start the Broker

**Option A: Docker (zero-install)**

```bash
docker-compose up -d
# Dashboard: http://127.0.0.1:8000
```

**Option B: Local Python (3.10+)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python broker.py --daemon
# Dashboard: http://127.0.0.1:8000
```



### Step 2: Connect Your Workers

**Local worker (same machine)**

```bash
HPO_BROKER_URL=http://localhost:8000 HPO_STUDY_NAME=my_study python train.py
```

**Remote worker (Colab / cloud GPU)**

To expose your local broker to remote workers, use a tunnel:

```bash
# Generate a token
export HPO_SECRET_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Start broker with tunnel (ngrok auto-generates a URL)
python broker.py --daemon --tunnel

# Or with Cloudflare (bring your own domain):
python broker.py --daemon --tunnel-provider cloudflare --tunnel-url https://your-domain.com
```

Set the printed URL and token on your remote machine:

```bash
export HPO_BROKER_URL="https://..."
export HPO_SECRET_TOKEN="<your-token>"
python train.py
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for more tunneling and auth options.

### Step 3: Agent Integration (optional)

Point your IDE at the MCP server for agent-driven onboarding and inspection. See [IDE Setup](#ide-setup-agent-driven-onboarding--inspection).

### Environment Variables Reference


| Variable               | Default                    | Description                                                                                                 |
| ---------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `HPO_DATABASE_URL`     | `sqlite:///hpo_studies.db` | SQLite connection string                                                                                    |
| `HPO_BROKER_URL`       | `http://localhost:8000`    | URL where the broker is running (required)                                                                  |
| `HPO_STUDY_NAME`       | *(none)*                   | Default study name when not passed explicitly                                                               |
| `HPO_SECRET_TOKEN`     | *(none)*                   | Bearer token for endpoints in remote deployments                                                            |
| `HPO_DEBUG`            | `0`                        | Set to `1` to enable verbose debug logging                                                                  |
| `HPO_SPARKLINES`       | `0`                        | Set to `1` to print a neat performance curve on trial completion :)                                         |
| `HPO_BACKUP_ON_START`  | `0`                        | Set to `1` to run a database backup when the broker starts.                                                 |
| `HPO_CAPTURE_FULL_ENV` | `0`                        | Set to `1` to capture all installed packages; default captures only whitelisted core framework dependencies |
| `HPO_TUNNEL_PROVIDER`  | *(none)*                   | Tunnel provider for remote access: `ngrok` or `cloudflare`                                                  |
| `HPO_TUNNEL_URL`       | *(none)*                   | Static tunnel URL when using `cloudflare` provider                                                          |
| `HPO_ALLOWED_ORIGINS`  | *(none)*                   | Additional CORS origins for the dashboard                                                                   |


---



## Core Features



### Optuna Engine

- **Tree-structured Parzen Estimator Sampler**: Probability based hyperparameter suggestions that beat grid and random search
- **Median Pruning**: Cuts underperforming trials early to save GPU time
- **Single or Dual-Objective**: Optimize one target, or map a Pareto front between a maximize and a minimize metric (e.g., accuracy vs. loss)
- **fANOVA Importances**: Identifies which hyperparameters actually matter



### Study Health Monitoring

The dashboard and `.hpo_status.json` show a health tier:


| Tier        | Meaning                                                 |
| ----------- | ------------------------------------------------------- |
| `healthy`   | Trials are completing, metrics are improving            |
| `watch`     | Stagnation or early warning signs                       |
| `intervene` | High OOM rate, prolonged stagnation, or 100% prune rate |


Health checks detect stagnation (flatlining score, loss) and hardware failure patterns (CUDA OOM on specific batch sizes).

### Persistent SQLite State

All configuration, trials, reviews, and metadata live in `hpo_studies.db`:

- Active search space and HPO config
- Trial results with VRAM telemetry
- Review history
- Generated model cards



### MCP Server

An MCP server (`hpo_mcp_server.py`) exposes structured study data through Model Context Protocol tools so your IDE agent can read study state, validate manifests, and register new studies.

---



## Agent Integration

Pathfinder exposes MCP tools that let your IDE agent (Cursor, Claude Code, Antigravity) participate in two workflows:

### Onboarding Flow

1. Agent reads your training script, identifies tunable hyperparameters and metrics
2. Agent drafts a `train.hpo.yaml` manifest
3. Agent calls `validate_manifest` to check for errors
4. Agent calls `init_from_manifest` to register the study in Optuna and SQLite
5. Agent writes a minimal worker script from `templates/worker_minimal.py`



### Inspection Flow

1. Agent calls `get_study_data` to retrieve trial telemetry, health tier, fANOVA importances, and best trials
2. Agent summarizes: current best score, health status, OOM rate, stagnation warnings
3. Search space adjustments happen through the dashboard Settings UI or `hpo_cli.py`

Key MCP tools: `validate_manifest`, `init_from_manifest`, `get_study_data`, `get_study_cards`, `export_manifest`.

Trigger phrases: say **"integrate HPO"** or **"wire hyperparameter tuning"** and your agent will walk through the onboarding flow. For inspection, say **"show study health"** or **"check HPO progress."**

See [AGENTS.md](AGENTS.md) for the full agent procedure.

---



## Onboarding Your Own Project



### 1. Write a manifest (`train.hpo.yaml`): Manually or have an agent do it for you

```yaml
study_name: my_study
metrics:
  primary_score: accuracy
  objectives:
    - name: accuracy
      direction: maximize
      label: "Accuracy"
    - name: loss
      direction: minimize
      label: "Loss"
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

The `primary_score` field tells the dashboard which objective to highlight.

### 2. Register the study

```bash
python hpo_cli.py validate train.hpo.yaml
python hpo_cli.py init train.hpo.yaml
```



### 3. Update your training script

```python
from src.hpo_client import TrialSession

session = TrialSession()  # reads HPO_BROKER_URL / HPO_STUDY_NAME
trial = session.suggest()
params = trial["params"]

for epoch in range(epochs):
    accuracy, loss = train_one_epoch(params, epoch)

    if session.report_epoch(epoch, score=accuracy, loss=loss):
        # Trial was pruned by the broker
        session.complete(epoch, score=accuracy, loss=loss, state="PRUNED")
        break

session.complete(epoch, score=accuracy, loss=loss, state="COMPLETE")
```

The worker contract is three calls: `suggest()`, `report_epoch()`, `complete()`. Map your higher-is-better metric to `score` and your lower-is-better metric to `loss`. Full details: [docs/INTEGRATION.md](docs/INTEGRATION.md).

### 4. Run on your GPU

```bash
export HPO_BROKER_URL=http://localhost:8000
export HPO_STUDY_NAME=my_study
python train.py
```

---



## IDE Setup (Agent-Driven Onboarding & Inspection)



### Cursor

**Settings → Features → MCP → + Add New MCP Server**

Name: `pathfinder`, Type: `command`, Command: `source .venv/bin/activate && python3 hpo_mcp_server.py`

### Claude Code / Antigravity

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

Pathfinder is compliant with the Model Context Protocol standard, it works with any MCP-compatible IDE.

---



## Common Commands

```bash
# Start broker + dashboard
python broker.py --daemon

# Validate and initialize a study from manifest
python hpo_cli.py validate train.hpo.yaml
python hpo_cli.py init train.hpo.yaml

# Check study health
python hpo_cli.py status

# Export study config back to YAML
python hpo_cli.py manifest my_study

# Export/import study data
python hpo_cli.py export my_study --output my_study.json
python hpo_cli.py import my_study.json

# Generate a model card
python hpo_cli.py modelcard my_study

# Delete a study
python hpo_cli.py delete my_study

# Run tests
pytest tests/ -q
```

---



## Limitations

Pathfinder runs on a single machine with SQLite. It does not support Postgres backends or advanced samplers like MOTPE or CMA-ES. This is a demonstration of MCP/agent integration for ML experiment workflows.

## What I Learned

- MCP tool design to inspect telemetry and modify training scripts, refactored the architecture to decouple agentic workflows from deterministic optimization path
- Implementing concurrency patterns for distributed workers, real-time detection of crashed processes
- Optimizing SQLite backend performance using Write-Ahead Logging; allowing concurrent broker writes, dashboard rendering, and MCP queries without read-write blocks

---



## Reference: crack-seg

Pathfinder was initially built to tune [crack-seg](https://github.com/Ishaan1402/crack-seg#crack-seg), a U-Net pixel-level segmentation model trained on UAV bridge imagery.

---



## Docs

- **[AGENTS.md](AGENTS.md)** — Guide for AI agents (Cursor, Claude Code, Antigravity, etc)
- **[docs/INTEGRATION.md](docs/INTEGRATION.md)** — Worker integration details

---



## License

MIT License - see the [LICENSE](LICENSE) file for details.