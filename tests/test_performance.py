"""Performance benchmarks — backs the README claim: "Suggests hyperparameters in <10ms."

Benchmarks two paths:
  1. Pure TPE sampling via optuna.study.ask() — the core suggestion engine.
  2. Full broker suggest via handle_api_suggest_trial — includes lease management, search-space
     resolution, and categorical repair, so it carries additional bookkeeping beyond the
     TPE call itself.  The <10ms claim refers to (1); (2) is included for completeness.
"""

import statistics
import time
import uuid
import optuna

from src.db_manager import DATABASE_URL


MAX_P50_MS = 10


def _init_study_with_trials(num_complete: int = 50) -> str:
    """Create a new study, suggest + complete N trials so TPE has historical data to work with."""
    from src.onboarding import initialize_study

    study_name = f"perf_{uuid.uuid4().hex[:12]}"
    space = {
        "learning_rate": {"min": 1e-5, "max": 1e-2, "type": "float_log"},
        "batch_size": {"options": [2, 4, 8, 16], "active": [2, 4, 8, 16], "type": "categorical"},
        "resolution": {"options": [256, 512, 1024], "active": [256, 512, 1024], "type": "categorical"},
        "loss_weight_ratio": {"min": 0.0, "max": 1.0, "type": "float"},
    }
    config = {"metric_score_label": "Score", "metric_loss_label": "Loss", "eval_protocol": {"enabled": False}}
    ctx = {"hypothesis": "perf benchmark", "gpu_model": "CPU", "gpu_capacity_gb": 8.0}
    initialize_study(study_name, space, config, ctx, multi_objective=True)

    study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)

    for _ in range(num_complete):
        trial = study.ask()
        study.tell(trial.number, values=[random_loss(), random_score()])

    return study_name


def random_loss() -> float:
    import random
    return round(random.uniform(0.05, 0.95), 4)


def random_score() -> float:
    import random
    return round(random.uniform(0.55, 0.99), 4)


# ---------------------------------------------------------------------------
# Benchmark 1 — pure TPE sampling
# ---------------------------------------------------------------------------

def test_tpe_suggest_latency():
    """Assert p50 of optuna study.ask() is under the README-claimed 10ms threshold."""
    study_name = _init_study_with_trials(50)
    study = optuna.load_study(study_name=study_name, storage=DATABASE_URL)

    # Warmup — let the sampler finish any JIT / one-off allocations
    for _ in range(50):
        trial = study.ask()
        study.tell(trial.number, values=[random_loss(), random_score()])

    timings_ms: list[float] = []
    for _ in range(200):
        start = time.perf_counter_ns()
        trial = study.ask()
        elapsed_ns = time.perf_counter_ns() - start
        timings_ms.append(elapsed_ns / 1_000_000)
        # Avoid accumulating unlimited trials in memory
        study.tell(trial.number, values=[random_loss(), random_score()])

    p50 = statistics.median(timings_ms)
    p95 = sorted(timings_ms)[int(len(timings_ms) * 0.95)]
    p99 = sorted(timings_ms)[int(len(timings_ms) * 0.99)]

    print(
        f"\n  TPE ask() — {len(timings_ms):,} samples after warmup\n"
        f"    p50 = {p50:.2f} ms   p95 = {p95:.2f} ms   p99 = {p99:.2f} ms"
    )

    assert p50 < MAX_P50_MS, (
        f"TPE p50 latency {p50:.2f} ms exceeds {MAX_P50_MS} ms threshold.\n"
        f"Either the benchmark environment is noisy or the claim needs updating in README."
    )


# ---------------------------------------------------------------------------
# Benchmark 2 — full broker suggest path
# ---------------------------------------------------------------------------

def test_broker_suggest_latency():
    """Full handle_api_suggest_trial wall time. Not bound by the 10ms README claim, but tracked."""
    from src.suggest import SuggestRequest, handle_api_suggest_trial

    study_name = _init_study_with_trials(50)

    # Warmup
    for _ in range(20):
        req = SuggestRequest(study_name=study_name, worker_id="perf-w")
        handle_api_suggest_trial(req)

    timings_ms: list[float] = []
    for _ in range(100):
        req = SuggestRequest(study_name=study_name, worker_id="perf-w")
        start = time.perf_counter_ns()
        handle_api_suggest_trial(req)
        elapsed_ns = time.perf_counter_ns() - start
        timings_ms.append(elapsed_ns / 1_000_000)

    p50 = statistics.median(timings_ms)
    p95 = sorted(timings_ms)[int(len(timings_ms) * 0.95)]

    print(
        f"\n  Broker full suggest — {len(timings_ms):,} samples after warmup\n"
        f"    p50 = {p50:.2f} ms   p95 = {p95:.2f} ms"
    )
