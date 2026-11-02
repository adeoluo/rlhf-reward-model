"""A deterministic PPO-vs-DPO comparison on the same synthetic preference task.

PPO (reward model + clipped policy gradient) and DPO (direct preference optimization, no
reward model) are given the same reference, the same preferences, and the same full-batch
optimizer. This script prints the final-quality comparison (true reward, gap closed, KL) and a
sample-efficiency table (true reward vs. the number of preference pairs). Deterministic and
offline; run with:

    uv run python benchmarks/compare_methods.py
"""

from __future__ import annotations

import time

from rlhf_reward_model.comparison import compare_ppo_dpo


def main() -> None:
    start = time.perf_counter()
    comparison = compare_ppo_dpo()
    elapsed = time.perf_counter() - start

    print("PPO vs DPO on the same task (reference, preferences, and optimizer shared):")
    print(
        f"  reference true reward {comparison.reference_reward:.4f}, "
        f"optimal {comparison.optimal_reward:.4f}"
    )
    print("  method   true reward   gap closed   mean KL")
    for result in (comparison.ppo, comparison.dpo):
        print(
            f"  {result.name:<7}  {result.true_reward:>10.4f}   "
            f"{result.reward_gap_closed:>9.1%}   {result.mean_kl_to_reference:>6.2f}"
        )

    print("\n  sample efficiency (true reward vs. number of preference pairs):")
    print("  preferences    PPO       DPO")
    for row in comparison.sample_efficiency:
        print(
            f"  {row.preferences:>11}   {row.ppo_true_reward:>7.4f}   {row.dpo_true_reward:>7.4f}"
        )
    print(f"\n  ({elapsed:.1f}s, deterministic)")


if __name__ == "__main__":
    main()
