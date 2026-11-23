# Evaluation Report - RLHF Reward Model

> Reproduce with `python -m rlhf_reward_model evaluate` (report) and
> `uv run python benchmarks/benchmark.py` (report, over-optimization study, and timings).
> Evaluation is deterministic: the task, the data, and all three stages are fully seeded.

## Setup
- **Task:** 6 prompts, 5 responses each, 12-dimensional features, hidden gold reward.
- **Reference:** behavior-cloned from 600 low-temperature demonstrations.
- **Reward model:** Bradley-Terry logistic regression on 4000 preferences (full-batch).
- **PPO:** 60 iterations, batch 256, 4 epochs per batch, clip 0.2, beta 0.05.
- **Held-out:** 800 preferences from a disjoint seed.

## Reward model

| Metric | Value |
|---|---:|
| true-reward correlation (Pearson) | 0.998 |
| held-out ranking accuracy | 0.984 |
| reward-model ceiling (true reward of its preferred responses) | 1.144 |

Because the true reward is linear in the features and the Bradley-Terry objective is concave,
full-batch gradient ascent recovers the reward almost exactly, so the reward-model ceiling
equals the true optimum.

## Policy

| Metric | Value |
|---|---:|
| reference true reward | 0.589 |
| PPO true reward | 1.125 |
| optimal true reward | 1.144 |
| reward gap closed | 96.6% |
| mean KL to reference | 1.673 |

## Reward over-optimization

A reward model restricted to 4 of the 12 features is a faithful proxy only in part. Optimizing
it at decreasing KL penalty traces the classic Goodhart curve:

| beta | proxy (reward-model) reward | true reward | mean KL |
|---:|---:|---:|---:|
| 0.20 | 0.238 | 0.824 | 0.92 |
| 0.10 | 0.340 | 0.938 | 1.60 |
| 0.05 | 0.414 | 0.934 | 2.60 |
| 0.02 | 0.458 | 0.859 | 3.82 |
| 0.01 | 0.458 | 0.858 | 3.83 |

These figures are from Linux x86-64, the platform CI checks them on. On Apple Silicon the first
two rows come out as `0.233 / 0.846 / 0.89` and `0.340 / 0.940 / 1.59`, because the linear-algebra
library rounds differently and sixty PPO iterations amplify it; the other rows and the shape of
the curve are the same. An earlier version of this table quoted the Apple values without saying
so, which is why a rerun on Linux did not match it.

The proxy reward rises monotonically while the true reward peaks at beta 0.1 and then declines.
The KL budget, not the proxy score, is the quantity to watch when the reward model is imperfect.

## Throughput
Supervised fine-tuning takes roughly 0.2 seconds, the reward model roughly 4 seconds, and PPO
roughly 1 second on an Apple M1 (`uv run python benchmarks/benchmark.py`). The project has no
service; an earlier version of this report quoted the throughput of one that was since removed.
