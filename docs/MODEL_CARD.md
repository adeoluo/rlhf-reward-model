# Model / System Card - RLHF Reward Model

## Overview
- **What it is:** a from-scratch three-stage RLHF pipeline (behavior-cloned reference,
  Bradley-Terry reward model, clipped PPO) over log-linear policies, with zero runtime
  dependencies and an explicit reward over-optimization study.
- **Base model(s):** from-scratch; no pretrained components.
- **Version and date:** 1.0, 2026-01-01.

## Intended use
- **Primary use cases:** teaching and demonstrating reward modeling, PPO, the KL-regularized
  objective, and reward over-optimization; a reference implementation for experiments on beta
  and reward-model specification.
- **Primary users:** engineers and researchers studying post-training and alignment.

## Out-of-scope / prohibited use
- Aligning a real language model. The policy is a log-linear contextual bandit, not a sequence
  model, so results transfer in spirit rather than scale.

## Performance
- Reward model true-reward correlation **0.998** and held-out accuracy **0.984**; PPO closes
  **96.6%** of the true reward gap from the reference to the optimum, with mean KL **1.673**.
  See [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Risks and limitations
- A misspecified reward model leads to reward over-optimization; the true reward peaks and then
  falls as the proxy is optimized harder. This is demonstrated rather than hidden.
- The Bradley-Terry annotator is more consistent than real human labeling.
- PPO optimizes a bandit, so there is no temporal credit assignment.

## Safety mitigations
- The HTTP service is read-only and validates that the prompt index is an integer in range
  before any computation runs.

## Environmental / cost footprint
- The full pipeline trains in roughly five seconds of single-core CPU time with no accelerator,
  so the energy footprint is negligible.
