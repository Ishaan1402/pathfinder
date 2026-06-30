# Archive

This directory contains code that was part of Pathfinder's development history but is not part of the current API surface.

## colab_worker.py

The original bridge-crack U-Net reference implementation. This was the study Pathfinder was initially built to tune — a pixel-level crack segmentation model trained on high-res UAV bridge imagery.

It still works as a standalone Colab worker, but uses domain-specific naming (dice/bce) that predates Pathfinder's generic score/loss metric abstraction. It is kept here for historical context and as evidence of the project's origins.

**Do not use this as a template for new studies.** Use `templates/worker_minimal.py` instead.
