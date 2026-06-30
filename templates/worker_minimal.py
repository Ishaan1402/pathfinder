"""Minimal HPO worker template.

Copy this next to your training code, fill in ``train_one_epoch``, and run it on your GPU box
(Colab, a server, anywhere). It talks to the broker only through ``hpo_client``.

Setup:
    export HPO_BROKER_URL="https://hpo.mycustomdomain.com"
    export HPO_STUDY_NAME="my_study"
    python worker_minimal.py
"""
from src.hpo_client import TrialSession

NUM_EPOCHS = 15


def train_one_epoch(params: dict, epoch: int) -> tuple[float, float]:
    """Run one epoch with the given hyperparameters and return (score, loss).

    Replace this body with your real training/validation step. `params` contains the
    hyperparameters the broker suggested (keys match the search space defined in your manifest).

    NOTE: The return value should be a tuple of (higher_is_better_score, lower_is_better_loss).
    The parameter names 'score' and 'loss' inside report_epoch are generalized:
      - 'score' maps to any higher-is-better metric (e.g. Accuracy, Dice, BLEU, Reward)
      - 'loss' maps to any lower-is-better loss (e.g. Cross-Entropy, BCE, MSE, L1, Perplexity)
    """
    raise NotImplementedError("Fill in train_one_epoch: train on params, return (score, loss).")


def main():
    # Detect GPU hardware telemetry if torch is available
    gpu_model = "CPU"
    max_vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            max_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass

    session = TrialSession()  # reads HPO_BROKER_URL / HPO_STUDY_NAME
    print(session.health())

    trial = session.suggest()
    print(f"Trial #{trial.get('trial_number')} params: {trial['params']}")

    last_epoch = 0
    score, loss = 0.0, 0.0
    pruned = False
    oom_triggered = False

    try:
        for epoch in range(NUM_EPOCHS):
            last_epoch = epoch
            score, loss = train_one_epoch(trial["params"], epoch)
            if session.report_epoch(epoch, score=score, loss=loss):
                session.complete(
                    epoch, score=score, loss=loss, state="PRUNED",
                    gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
                )
                pruned = True
                print(f"Trial pruned at epoch {epoch}.")
                break
    except Exception as exc:
        # Catch Out Of Memory error (robust check across PyTorch versions / plain exception strings)
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "oom" in exc_str:
            oom_triggered = True
            print(f"OOM triggered during trial execution: {exc}")
            session.complete(
                last_epoch, score=score, loss=loss, state="FAIL",
                gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=True
            )
            print("Trial failed due to GPU OOM. Continuing to next trial.")
            return
        else:
            # Re-raise standard training exceptions
            raise exc

    if not pruned and not oom_triggered:
        session.complete(
            last_epoch, score=score, loss=loss, weights_path="model.pt", state="COMPLETE",
            gpu_model=gpu_model, max_vram_gb=max_vram_gb, oom_triggered=False
        )
        print("Trial complete.")


if __name__ == "__main__":
    main()
