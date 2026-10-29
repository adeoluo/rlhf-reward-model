"""The alignment task: prompts, candidate responses, and a hidden gold reward.

Direct Preference Optimization is studied here on a tractable contextual bandit so
that every quantity is exact and inspectable. There are a handful of prompts, each
with a small set of candidate responses. Every (prompt, response) pair has a fixed
feature vector, and a hidden gold weight vector defines the *true* reward as a dot
product. Human preferences are simulated by a Bradley-Terry annotator that prefers
the response with the higher true reward, but only probabilistically, so the
preference data is realistically noisy.

This mirrors the real post-training pipeline: a supervised policy is fit to
demonstrations, a preference dataset is collected, and preference optimization
aligns the policy, all while the true reward stays hidden from the learner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Preference:
    """A single annotated comparison for one prompt."""

    prompt: int
    chosen: int
    rejected: int


@dataclass(frozen=True)
class Demonstration:
    """A supervised (prompt, response) target for behavior cloning."""

    prompt: int
    response: int


@dataclass(frozen=True)
class Domain:
    """A fixed alignment task with a hidden gold reward over prompt-response pairs."""

    n_prompts: int
    n_responses: int
    dim: int
    features: list[list[list[float]]]
    gold_weight: list[float]

    @classmethod
    def build(
        cls, *, n_prompts: int = 6, n_responses: int = 5, dim: int = 12, seed: int = 0
    ) -> Domain:
        rng = random.Random(seed)
        features = [
            [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n_responses)]
            for _ in range(n_prompts)
        ]
        gold_weight = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        return cls(
            n_prompts=n_prompts,
            n_responses=n_responses,
            dim=dim,
            features=features,
            gold_weight=gold_weight,
        )

    def feature(self, prompt: int, response: int) -> list[float]:
        return self.features[prompt][response]

    def true_reward(self, prompt: int, response: int) -> float:
        phi = self.features[prompt][response]
        return sum(w * x for w, x in zip(self.gold_weight, phi, strict=True)) / math.sqrt(self.dim)

    def best_response(self, prompt: int) -> int:
        return max(range(self.n_responses), key=lambda r: self.true_reward(prompt, r))


def _bradley_terry(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin))


def demonstrations(
    domain: Domain, count: int, *, temperature: float, seed: int
) -> list[Demonstration]:
    """Sample weakly aligned supervised targets from a low-temperature teacher.

    The teacher samples responses in proportion to ``exp(temperature * reward)``.
    A small temperature yields a mediocre, only partly aligned behavior policy, which
    is exactly the kind of supervised data a reference model is fit to.
    """

    rng = random.Random(seed)
    samples: list[Demonstration] = []
    for _ in range(count):
        prompt = rng.randrange(domain.n_prompts)
        logits = [temperature * domain.true_reward(prompt, r) for r in range(domain.n_responses)]
        high = max(logits)
        weights = [math.exp(value - high) for value in logits]
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = domain.n_responses - 1
        for response, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen = response
                break
        samples.append(Demonstration(prompt=prompt, response=chosen))
    return samples


def preferences(domain: Domain, count: int, *, seed: int) -> list[Preference]:
    """Sample noisy pairwise preferences from a Bradley-Terry annotator."""

    rng = random.Random(seed)
    data: list[Preference] = []
    for _ in range(count):
        prompt = rng.randrange(domain.n_prompts)
        first, second = rng.sample(range(domain.n_responses), 2)
        margin = domain.true_reward(prompt, first) - domain.true_reward(prompt, second)
        if rng.random() < _bradley_terry(margin):
            data.append(Preference(prompt=prompt, chosen=first, rejected=second))
        else:
            data.append(Preference(prompt=prompt, chosen=second, rejected=first))
    return data
