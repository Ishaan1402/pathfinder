# Pathfinder Onboarding Walkthrough

This directory contains a complete, self-contained walkthrough for onboarding a new ML project to Pathfinder using a **manifest file** (`train.hpo.yaml`).

---

## The Happy Path (3 Steps)

### Step 1: Draft the Manifest
Create a file named `train.hpo.yaml` defining your search space, objective metrics, environment variables, and entrypoint command. You can start from [train.hpo.yaml](train.hpo.yaml).

> [!NOTE]
> **Worker Metrics Slot Mapping**
> Even if you name your objectives `"accuracy"` (maximize) and `"loss"` (minimize) in the manifest, the worker client script will report them through the generic API slots: `report_epoch(..., score=..., loss=...)`. Pathfinder maps them dynamically in the backend using your objective directions.

### Step 2: Validate the Manifest
Validate your manifest file before registering:
- **CLI**:
  ```bash
  python hpo_cli.py validate examples/onboarding/train.hpo.yaml
  ```
- **Dashboard**: Open the dashboard, click **+ New Study** in the top right, and drag-and-drop your `train.hpo.yaml` file into the modal.
- **MCP**: Call the `validate_manifest` tool with your YAML content.

### Step 3: Initialize the Study
Register your study and write configurations to the SQLite database:
- **CLI**:
  ```bash
  python hpo_cli.py init examples/onboarding/train.hpo.yaml
  ```
- **Dashboard**: Click **Initialize Study** in the New Study modal once validation succeeds.
- **MCP**: Call the `init_from_manifest` tool.

---

## Running the Worker

Once the study is initialized:
1. Open the dashboard.
2. Select your study (`mnist_tuning`) in the dropdown.
3. Click the **Worker Setup** tab in the sidebar.
4. Copy the fully customized setup commands and run them on your training GPU machine or in a Google Colab notebook!
