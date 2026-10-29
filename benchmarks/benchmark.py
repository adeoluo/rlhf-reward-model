"""A deterministic micro-benchmark for the RLHF pipeline.

Reports the three training stage timings and the headline metrics, then
demonstrates reward over-optimization (Goodhart's law): a deliberately
misspecified reward model is optimized at decreasing KL penalty, and the proxy
(reward-model) reward keeps rising while the true reward peaks and then declines.
Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import evaluate, expected_reward
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModel, RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


def _expected_proxy_reward(domain: Domain, policy: SoftmaxPolicy, model: RewardModel) -> float:
    total = 0.0
    for prompt in range(domain.n_prompts):
        distribution = policy.distribution(prompt)
        total += sum(
            probability * model.reward(prompt, response)
            for response, probability in enumerate(distribution)
        )
    return total / domain.n_prompts


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

    print("reward over-optimization with a misspecified reward model (4 of 12 features):")
    partial = train_reward_model(domain, train, RewardModelConfig(active_features=4, seed=5))
    print("  beta   proxy reward   true reward   mean KL")
    for beta in (0.2, 0.1, 0.05, 0.02, 0.01):
        hacked = train_ppo(domain, reference, partial, PPOConfig(beta=beta, seed=6))
        proxy = _expected_proxy_reward(domain, hacked, partial)
        true = expected_reward(domain, hacked)
        report = evaluate(domain, hacked, reference, partial, held_out)
        print(f"  {beta:<5}  {proxy:>11.4f}   {true:>10.4f}   {report.mean_kl_to_reference:>6.2f}")


if __name__ == "__main__":
    main()
