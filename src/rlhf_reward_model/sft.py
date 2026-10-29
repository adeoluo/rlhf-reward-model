"""Supervised fine-tuning by behavior cloning.

The reference policy is fit to demonstrations by maximum likelihood. The gradient
of the log-likelihood of a demonstration under a log-linear policy is the
demonstration feature vector minus the policy's expected feature vector, so a
single pass of stochastic gradient ascent with an L2 penalty suffices.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlhf_reward_model.domain import Demonstration, Domain
from rlhf_reward_model.policy import SoftmaxPolicy


@dataclass(frozen=True)
class SFTConfig:
    """Behavior-cloning hyperparameters."""

    epochs: int = 40
    learning_rate: float = 0.5
    l2: float = 1e-3
    seed: int = 0


def behavior_clone(domain: Domain, data: list[Demonstration], config: SFTConfig) -> SoftmaxPolicy:
    """Fit a reference policy to demonstrations by penalized maximum likelihood."""

    import random

    policy = SoftmaxPolicy(domain=domain)
    rng = random.Random(config.seed)
    order = list(data)
    for _ in range(config.epochs):
        rng.shuffle(order)
        for demo in order:
            phi = domain.feature(demo.prompt, demo.response)
            expected = policy.expected_features(demo.prompt)
            for index in range(domain.dim):
                gradient = phi[index] - expected[index] - config.l2 * policy.weight[index]
                policy.weight[index] += config.learning_rate * gradient
    return policy
