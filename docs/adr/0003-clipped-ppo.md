# ADR-0003: A clipped PPO surrogate with a KL-shaped reward

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The policy-optimization stage must raise the learned reward without drifting far from the
reference, and it must be stable enough to run without a tensor library or automatic
differentiation. Vanilla policy gradients are high variance and can take destructive steps; a
trust-region method is the standard remedy.

## Decision
Use the **PPO clipped surrogate** over the log-linear policy. The KL penalty is folded into the
reward, `shaped = reward_model(y) - beta * (log policy_old(y) - log reference(y))`, so the
advantage already carries the regularization. Each iteration samples responses from the current
policy, estimates a per-prompt mean baseline from those samples, and takes several clipped
gradient steps with the exact policy-gradient term `phi(y) - E_policy[phi]`.

## Consequences
- Clipping bounds each update so a single batch cannot move the policy arbitrarily far, which is
  what makes the reinforcement stage stable in pure Python.
- Folding the KL penalty into the reward keeps the objective a standard advantage-weighted
  surrogate, and beta is the single knob that trades reward for divergence.
- The baseline is a per-prompt mean rather than a learned critic, which suffices for a bandit;
  a temporal task would need generalized advantage estimation and a value function, recorded as
  a roadmap item.
