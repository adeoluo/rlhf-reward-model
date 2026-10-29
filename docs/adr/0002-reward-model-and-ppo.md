# ADR-0002: A reward model and PPO, not Direct Preference Optimization

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
There are two dominant ways to turn human preferences into an aligned policy. Direct Preference
Optimization folds the reward into the policy and fits it with a single classification loss on
comparisons, avoiding a separate reward model and any reinforcement learning. Classical RLHF
instead trains an explicit reward model and then optimizes the policy against it with a clipped
policy gradient and a KL penalty. A companion project in this portfolio implements DPO; this one
implements the classical pipeline on purpose.

## Decision
Implement the full **three-stage RLHF** pipeline: supervised fine-tuning by behavior cloning, a
**Bradley-Terry reward model** fit by convex logistic regression, and **PPO** against the
learned reward with a KL penalty. Keep the reward model explicit and inspectable.

## Consequences
- The reward model is a first-class artifact: its correlation with the true reward and its
  held-out accuracy can be measured directly, and it can be deliberately misspecified to study
  failure.
- Because the policy optimizes a *proxy* reward, the pipeline can exhibit **reward
  over-optimization**, where optimizing the proxy harder eventually lowers the true reward. The
  benchmark reproduces this curve, which a DPO-only treatment cannot show as cleanly.
- The cost is more moving parts than DPO (a reward model plus a reinforcement stage), and PPO
  needs its own stabilization (clipping, a baseline, and the KL penalty).
