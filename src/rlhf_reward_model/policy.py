"""A log-linear softmax policy over candidate responses.

The policy scores a response by the dot product of a weight vector with the
(prompt, response) feature vector and normalizes over the candidates for a prompt.
Because the model is log-linear, the log-probability, the full distribution, the
expected feature vector, and every gradient used by supervised fine-tuning and by
Direct Preference Optimization are available in closed form.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from rlhf_reward_model.domain import Domain


def _logsumexp(values: list[float]) -> float:
    high = max(values)
    return high + math.log(sum(math.exp(value - high) for value in values))


@dataclass
class SoftmaxPolicy:
    """Weights over response features, with exact distributions and gradients."""

    domain: Domain
    weight: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weight:
            self.weight = [0.0] * self.domain.dim

    def copy(self) -> SoftmaxPolicy:
        return SoftmaxPolicy(domain=self.domain, weight=list(self.weight))

    def _logits(self, prompt: int) -> list[float]:
        return [
            sum(w * x for w, x in zip(self.weight, self.domain.feature(prompt, r), strict=True))
            for r in range(self.domain.n_responses)
        ]

    def distribution(self, prompt: int) -> list[float]:
        logits = self._logits(prompt)
        normalizer = _logsumexp(logits)
        return [math.exp(value - normalizer) for value in logits]

    def log_prob(self, prompt: int, response: int) -> float:
        logits = self._logits(prompt)
        return logits[response] - _logsumexp(logits)

    def expected_features(self, prompt: int) -> list[float]:
        distribution = self.distribution(prompt)
        expected = [0.0] * self.domain.dim
        for response, probability in enumerate(distribution):
            phi = self.domain.feature(prompt, response)
            for index in range(self.domain.dim):
                expected[index] += probability * phi[index]
        return expected

    def sample(self, prompt: int, rng: random.Random) -> int:
        distribution = self.distribution(prompt)
        threshold = rng.random()
        cumulative = 0.0
        for response, probability in enumerate(distribution):
            cumulative += probability
            if cumulative >= threshold:
                return response
        return self.domain.n_responses - 1

    def greedy(self, prompt: int) -> int:
        distribution = self.distribution(prompt)
        return max(range(self.domain.n_responses), key=lambda r: distribution[r])
