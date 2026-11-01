"""A large-workload scale benchmark for reward-model training.

Fits the Bradley-Terry reward model on progressively larger synthetic preference
datasets and reports, per workload, the wall-clock time, throughput in preference
pairs per second, the peak traced memory, and the held-out ranking accuracy. It
answers a different question from ``benchmark.py`` (which profiles the full
pipeline once): how the core fitting operation scales as the preference dataset
grows.

Timing and memory are captured in separate phases on purpose. ``tracemalloc``
instruments every allocation and dominates the wall-clock of these tight numeric
loops, so it would distort the throughput numbers; the peak footprint is governed
by the resident preference dataset and the model, not by the iteration count, so a
short traced pass measures it faithfully. Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from rlhf_reward_model.domain import Domain, Preference, preferences
from rlhf_reward_model.evaluation import reward_model_accuracy
from rlhf_reward_model.reward_model import RewardModelConfig, train_reward_model

WORKLOADS: tuple[int, ...] = (2000, 8000, 16000)
_ITERATIONS = 120
_MEM_ITERATIONS = 3
_HELD_OUT = 800
_PREF_SEED = 3
_HELD_OUT_SEED = 4
_MODEL_SEED = 5


@dataclass(frozen=True)
class ScalePoint:
    """One row of the scale table for a single workload size."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    accuracy: float


def measure(
    domain: Domain,
    held_out: list[Preference],
    size: int,
    *,
    iterations: int = _ITERATIONS,
    mem_iterations: int = _MEM_ITERATIONS,
) -> ScalePoint:
    """Fit the reward model on ``size`` preference pairs and measure the cost."""

    tracemalloc.start()
    tracemalloc.reset_peak()
    data = preferences(domain, size, seed=_PREF_SEED)
    train_reward_model(domain, data, RewardModelConfig(iterations=mem_iterations, seed=_MODEL_SEED))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start = time.perf_counter()
    model = train_reward_model(
        domain, data, RewardModelConfig(iterations=iterations, seed=_MODEL_SEED)
    )
    seconds = time.perf_counter() - start

    return ScalePoint(
        size=size,
        seconds=seconds,
        throughput=size / seconds if seconds > 0 else 0.0,
        peak_mib=peak / 1048576,
        accuracy=reward_model_accuracy(domain, model, held_out),
    )


def main() -> None:
    print("Synthetic preference data; a pure-Python, from-scratch reward model.")
    print("Wall-clock is not competitive with compiled numerical libraries.")
    print(f"Fitting the Bradley-Terry reward model for {_ITERATIONS} full-batch iterations.\n")

    domain = Domain.build(seed=0)
    held_out = preferences(domain, _HELD_OUT, seed=_HELD_OUT_SEED)

    print(f"{'pairs':>8}  {'seconds':>8}  {'pairs/sec':>10}  {'peak MiB':>9}  {'accuracy':>8}")
    for size in WORKLOADS:
        point = measure(domain, held_out, size)
        print(
            f"{point.size:>8}  {point.seconds:>8.3f}  {point.throughput:>10.0f}  "
            f"{point.peak_mib:>9.2f}  {point.accuracy:>8.3f}"
        )


if __name__ == "__main__":
    main()
