# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development & Test Commands

- **Environment Setup**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Run FastAPI Broker Server**:
  ```bash
  python3 broker.py --daemon              # local only
  export HPO_SECRET_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  python3 broker.py --daemon --tunnel     # Colab / remote GPU (auth required)
  ```
- **Run Simulated Training Worker / GPU Runner**:
  ```bash
  # Standard worker simulation:
  python3 simulators/training_worker.py
  
  # Or run default study:
  HPO_BROKER_URL=http://localhost:8123 HPO_STUDY_NAME=seg_v1 python3 simulators/training_worker.py
  ```
- **Launch Visualization Dashboard (Optuna)**:
  ```bash
  optuna-dashboard sqlite:///hpo_studies.db
  ```
- **Run Integration Tests**:
  ```bash
  .venv/bin/python3 tests/test_integration.py
  ```

---

## High-Level Architecture

This project implements a decentralized **Worker-Broker-Registry** pattern for hyperparameter tuning, specifically optimized for a **U-Net crack segmentation project** (and adaptable via templates). It avoids blocking training workers on LLM calls.

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

### Architecture Components

1. **SQLite Database (`hpo_studies.db`)**: The **single source of truth** for all persistence. Holds both Optuna's trial states and custom metadata tables (`trial_results`, `system_configuration`, `study_reviews`, `study_status`, `agent_reasoning_logs`, `invalid_proposals`).
2. **FastAPI Broker (`broker.py`)**: Thin HTTP broker API exposing endpoints for workers (`/api/suggest_trial`, `/api/report_epoch`, `/api/complete_trial`) and the dashboard web interface (`/api/hpo_config`, `/api/review_packet`).
3. **MCP Server (`hpo_mcp_server.py`)**: FastMCP server exposing tools for human-in-the-loop coordination, onboarding, and reviews to IDE agents.
4. **Decoupled Worker (`src/hpo_client.py`, `colab_worker.py`, `simulators/training_worker.py`)**: Interacts exclusively via HTTP using the 3-step life cycle (`suggest` -> `report_epoch` -> `complete`). Colab reference: `train_colab_trial` (one trial) and `train_colab_trial_loop` (repeated session).
5. **Decoupled Evaluator (Interactive Dashboard)**: Custom dashboard calling the FastAPI routes to view trials, Pareto fonts, fANOVA parameter importances, and toggle coordinator reviews.

---

## Code Style & Development Guidelines

1. **State & DB Isolation**: State MUST live in SQLite. Do **not** write or persist temporary configurations to files on disk like `active_search_space.json` or `hpo_config.json`. Always load/save via the `SystemConfiguration` ORM table.
2. **Database Resilience**: Column additions or schema model changes should be registered in `src/db_manager.py:__ADDITIVE_COLUMNS` to handle additive, idempotent migrations on runtime initialization instead of dropping tables.
3. **Coordinator Reviews (Episodic LLM)**: IDE agents act as **episodic coordinators**. Optuna (TPE) is the hot sampling path; workers NEVER block on language models.
4. **Review Procedure**:
   - Retrieve compacted statistical packet with `get_study_data()`.
   - Perform a safety review of VRAM and evaluate health alerts/triggers.
   - Adjust active bounds or propose changes with `update_search_space()`.
   - Submit review idempotently via `submit_agent_review()`.
5. **Worker Integration Contract**: Ensure newly integrated training scripts use `TrialSession` client rather than direct SQL queries or custom database drivers.
