# Scaling - RLHF Reward Model

This note describes how the pipeline behaves as data and load grow, and the tradeoffs on the
path to a real RLHF system.

## Where the time goes
- **Supervised fine-tuning** is a maximum-likelihood fit over the demonstrations and finishes in
  a fraction of a second.
- **Reward model** training is full-batch gradient ascent over the preference pairs; each
  iteration is linear in the number of comparisons. It is the second most expensive stage.
- **PPO** samples a batch from the current policy each iteration, so its cost is the number of
  iterations times the batch size times a constant per sample.
- **Serving** is read-only: a score query evaluates two distributions and the rewards over a
  fixed, small response set, which is sub-millisecond.

## Request-time scaling
The service is stateless and the models are immutable after startup, so there is no shared
mutable state and no lock on the hot path. A stress benchmark
(`uv run python benchmarks/stress.py`) drives the score handler under concurrency and reports
over 27,000 requests per second with sub-millisecond median latency. Horizontal scaling is
adding replicas behind a load balancer.

## Data scaling
Reward-model training time is linear in the number of preferences and iterations; memory is the
single weight vector plus the task tensors. PPO cost grows with iterations and batch size, not
with the preference-set size.

## Tradeoffs on the path to a real system
- **Reward model quality is the ceiling.** The policy can only be as aligned as the reward
  model is faithful; a misspecified reward is over-optimized, as the evaluation shows. Reward
  model ensembles and uncertainty penalties are the standard mitigations.
- **Beta.** The KL penalty trades reward for divergence and, crucially, guards against
  over-optimization. It is the knob to tune, not the proxy score.
- **Bandit versus sequence model.** The log-linear bandit keeps everything exact. A real
  language-model policy needs per-token log-probabilities, a value critic, generalized advantage
  estimation, and automatic differentiation, trading transparency for scale.

## Limits
Training is pure Python and single-threaded, ideal for study and modest data but not for
web-scale corpora. The controlled task bounds the conclusions; scaling to real RLHF is a data,
systems, and modeling problem, not a throughput one.
