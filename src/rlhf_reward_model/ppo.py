"""PPO-style policy optimization against a learned reward.

This is the third stage of classical RLHF. A policy is optimized to maximize the
reward model's score while a KL penalty to the frozen reference keeps it from
drifting. The reward is shaped by folding the KL penalty into it,
``shaped = reward_model(y) - beta * (log policy_old(y) - log reference(y))``, and
the policy is updated with the clipped PPO surrogate on the resulting advantage.

Each iteration samples responses from the current policy (the behavior policy for
that iteration), estimates a per-prompt baseline from those samples, and takes
several clipped gradient steps. The gradient of ``log policy(y|x)`` for the
log-linear policy is the response feature vector minus the policy's expected
feature vector, so the whole update is exact and dependency-free.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from rlhf_reward_model.domain import Domain
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.reward_model import RewardModel


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters."""

    iterations: int = 60
    batch_size: int = 256
    epochs_per_batch: int = 4
    learning_rate: float = 0.05
    clip: float = 0.2
    beta: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class _Sample:
    prompt: int
    response: int
    advantage: float
    old_log_prob: float


def _collect(
    domain: Domain,
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    reward_model: RewardModel,
    config: PPOConfig,
    rng: random.Random,
) -> list[_Sample]:
    raw: list[tuple[int, int, float, float]] = []
    prompt_rewards: dict[int, list[float]] = {}
    for _ in range(config.batch_size):
        prompt = rng.randrange(domain.n_prompts)
        response = policy.sample(prompt, rng)
        old_log_prob = policy.log_prob(prompt, response)
        reference_log_prob = reference.log_prob(prompt, response)
        shaped = reward_model.reward(prompt, response) - config.beta * (
            old_log_prob - reference_log_prob
        )
        raw.append((prompt, response, shaped, old_log_prob))
        prompt_rewards.setdefault(prompt, []).append(shaped)
    baselines = {prompt: sum(values) / len(values) for prompt, values in prompt_rewards.items()}
    return [
        _Sample(
            prompt=prompt,
            response=response,
            advantage=shaped - baselines[prompt],
            old_log_prob=old_log_prob,
        )
        for prompt, response, shaped, old_log_prob in raw
    ]


def train_ppo(
    domain: Domain,
    reference: SoftmaxPolicy,
    reward_model: RewardModel,
    config: PPOConfig,
) -> SoftmaxPolicy:
    """Optimize a copy of the reference against the reward model with clipped PPO."""

    import math

    policy = reference.copy()
    rng = random.Random(config.seed)
    for _ in range(config.iterations):
        batch = _collect(domain, policy, reference, reward_model, config, rng)
        for _ in range(config.epochs_per_batch):
            for sample in batch:
                log_prob = policy.log_prob(sample.prompt, sample.response)
                ratio = math.exp(log_prob - sample.old_log_prob)
                advantage = sample.advantage
                clipped = max(1.0 - config.clip, min(1.0 + config.clip, ratio))
                if ratio * advantage <= clipped * advantage:
                    coefficient = advantage * ratio
                else:
                    continue
                phi = domain.feature(sample.prompt, sample.response)
                expected = policy.expected_features(sample.prompt)
                for index in range(domain.dim):
                    gradient = coefficient * (phi[index] - expected[index])
                    policy.weight[index] += config.learning_rate * gradient
    return policy
