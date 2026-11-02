"""A deterministic micro-benchmark for the RLHF pipeline.

Reports the three training stage timings and the headline metrics, then
demonstrates reward over-optimization (Goodhart's law) via the library's
``overoptimization_study``: a deliberately misspecified reward model is optimized
at decreasing KL penalty, and the proxy (reward-model) reward keeps rising while
the true reward peaks and then declines. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import evaluate
from rlhf_reward_model.overoptimization import overoptimization_study
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


def main() -> None:
    domain = Domain.build(seed=0)
    sft_data = demonstrations(domain, 600, temperature=1.0, seed=1)
    train = preferences(domain, 4000, seed=3)
    held_out = preferences(domain, 800, seed=4)

    start = time.perf_counter()
    reference = behavior_clone(domain, sft_data, SFTConfig(seed=2))
    sft_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    model = train_reward_model(domain, train, RewardModelConfig(seed=5))
    rm_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    policy = train_ppo(domain, reference, model, PPOConfig(beta=0.05, seed=6))
    ppo_ms = (time.perf_counter() - start) * 1000

    report = evaluate(domain, policy, reference, model, held_out)
    print(f"Supervised fine-tuning ({len(sft_data)} demos): {sft_ms:.0f} ms")
    print(f"Reward model ({len(train)} preferences): {rm_ms:.0f} ms")
    print(f"PPO (60 iterations): {ppo_ms:.0f} ms")
    print(f"  {report.format()}")

    study = overoptimization_study()
    print(
        f"reward over-optimization with a misspecified reward model "
        f"({study.active_features} of {study.total_features} features):"
    )
    print("  beta   proxy reward   true reward   mean KL")
    for row in study.rows:
        print(
            f"  {row.beta:<5}  {row.proxy_reward:>11.4f}   "
            f"{row.true_reward:>10.4f}   {row.mean_kl:>6.2f}"
        )
    print(f"  true reward peaks at beta={study.best_beta} ({study.best_true_reward:.4f}).")


if __name__ == "__main__":
    main()
