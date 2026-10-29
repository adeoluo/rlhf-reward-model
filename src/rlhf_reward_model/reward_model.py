"""A Bradley-Terry reward model fit from pairwise preferences.

Unlike Direct Preference Optimization, classical RLHF trains an explicit reward
model. The model scores a response by a linear function of its features, and the
probability that a chosen response beats a rejected one is the logistic of their
score difference. Fitting is penalized logistic regression on the feature
differences, with a hand-derived gradient. The learned reward is later optimized
by a policy, so its fidelity to the hidden true reward is what ultimately bounds
how well the policy can do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rlhf_reward_model.domain import Domain, Preference


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class RewardModelConfig:
    """Reward-model training hyperparameters (full-batch gradient descent).

    ``active_features`` optionally restricts the model to the first ``k`` feature
    dimensions, which produces a deliberately misspecified reward model that cannot
    represent the true reward. Optimizing such a model hard is how reward
    over-optimization (Goodhart's law) is demonstrated.
    """

    iterations: int = 400
    learning_rate: float = 1.0
    l2: float = 1e-3
    active_features: int | None = None
    seed: int = 0


@dataclass
class RewardModel:
    """A linear reward over response features, fit by Bradley-Terry logistic regression."""

    domain: Domain
    weight: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weight:
            self.weight = [0.0] * self.domain.dim

    def reward(self, prompt: int, response: int) -> float:
        phi = self.domain.feature(prompt, response)
        return sum(w * x for w, x in zip(self.weight, phi, strict=True))

    def ranks_correctly(self, prompt: int, better: int, worse: int) -> bool:
        return self.reward(prompt, better) > self.reward(prompt, worse)


def _log_likelihood(model: RewardModel, data: list[Preference]) -> float:
    total = 0.0
    for preference in data:
        margin = model.reward(preference.prompt, preference.chosen) - model.reward(
            preference.prompt, preference.rejected
        )
        total += math.log(_sigmoid(margin)) if margin > -700 else margin
    return total / len(data) if data else 0.0


def train_reward_model(
    domain: Domain, data: list[Preference], config: RewardModelConfig
) -> RewardModel:
    """Fit the reward model by penalized Bradley-Terry maximum likelihood.

    The Bradley-Terry log-likelihood is concave in the weights, so full-batch
    gradient ascent converges stably to the regularized optimum. Each iteration
    averages the per-pair gradient ``sigmoid(-margin) * (phi_chosen - phi_rejected)``
    over the dataset and applies an L2 shrinkage term.
    """

    model = RewardModel(domain=domain)
    count = len(data) or 1
    active = config.active_features if config.active_features is not None else domain.dim
    for _ in range(config.iterations):
        gradient = [0.0] * domain.dim
        for preference in data:
            phi_w = domain.feature(preference.prompt, preference.chosen)
            phi_l = domain.feature(preference.prompt, preference.rejected)
            margin = model.reward(preference.prompt, preference.chosen) - model.reward(
                preference.prompt, preference.rejected
            )
            scale = _sigmoid(-margin)
            for index in range(active):
                gradient[index] += scale * (phi_w[index] - phi_l[index])
        for index in range(active):
            averaged = gradient[index] / count - config.l2 * model.weight[index]
            model.weight[index] += config.learning_rate * averaged
    return model
