"""Evaluation for the RLHF pipeline: reward model fidelity and policy improvement.

The metrics separate two questions. First, how good is the learned reward model,
measured by its held-out ranking accuracy and its correlation with the hidden true
reward. Second, how good is the PPO-optimized policy, measured against the true
reward. Because the policy optimizes the *learned* reward, its ceiling is the true
reward of the reward model's preferred responses, so the gap between that ceiling
and the true optimum is exactly the cost of an imperfect reward model, which is the
seed of reward hacking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rlhf_reward_model.domain import Domain, Preference
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.reward_model import RewardModel


@dataclass(frozen=True)
class RLHFReport:
    """A summary of reward-model fidelity and policy improvement."""

    reward_model_accuracy: float
    reward_model_correlation: float
    reference_reward: float
    policy_reward: float
    optimal_reward: float
    reward_model_ceiling: float
    mean_kl_to_reference: float

    @property
    def reward_gap_closed(self) -> float:
        span = self.optimal_reward - self.reference_reward
        if span <= 0.0:
            return 0.0
        return (self.policy_reward - self.reference_reward) / span

    @property
    def reward_model_tax(self) -> float:
        return self.optimal_reward - self.reward_model_ceiling

    def format(self) -> str:
        return (
            f"reward model: accuracy {self.reward_model_accuracy:.3f}, "
            f"true-reward correlation {self.reward_model_correlation:.3f}  |  "
            f"policy true reward {self.policy_reward:.4f} "
            f"(reference {self.reference_reward:.4f}, optimal {self.optimal_reward:.4f}; "
            f"gap closed {self.reward_gap_closed:.1%})  "
            f"reward-model ceiling {self.reward_model_ceiling:.4f} "
            f"(tax {self.reward_model_tax:.4f})  "
            f"mean KL {self.mean_kl_to_reference:.3f}"
        )


def expected_reward(domain: Domain, policy: SoftmaxPolicy) -> float:
    total = 0.0
    for prompt in range(domain.n_prompts):
        distribution = policy.distribution(prompt)
        total += sum(
            probability * domain.true_reward(prompt, response)
            for response, probability in enumerate(distribution)
        )
    return total / domain.n_prompts


def optimal_reward(domain: Domain) -> float:
    return (
        sum(
            domain.true_reward(prompt, domain.best_response(prompt))
            for prompt in range(domain.n_prompts)
        )
        / domain.n_prompts
    )


def reward_model_ceiling(domain: Domain, model: RewardModel) -> float:
    """Average true reward of the response the reward model most prefers per prompt."""

    total = 0.0
    for prompt in range(domain.n_prompts):
        preferred = max(range(domain.n_responses), key=lambda r: model.reward(prompt, r))
        total += domain.true_reward(prompt, preferred)
    return total / domain.n_prompts


def reward_model_accuracy(domain: Domain, model: RewardModel, data: list[Preference]) -> float:
    if not data:
        return 0.0
    correct = 0
    for preference in data:
        better, worse = preference.chosen, preference.rejected
        if domain.true_reward(preference.prompt, worse) > domain.true_reward(
            preference.prompt, better
        ):
            better, worse = worse, better
        if model.ranks_correctly(preference.prompt, better, worse):
            correct += 1
    return correct / len(data)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0.0 or variance_y == 0.0:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def reward_model_correlation(domain: Domain, model: RewardModel) -> float:
    learned: list[float] = []
    true: list[float] = []
    for prompt in range(domain.n_prompts):
        for response in range(domain.n_responses):
            learned.append(model.reward(prompt, response))
            true.append(domain.true_reward(prompt, response))
    return _pearson(learned, true)


def mean_kl(domain: Domain, policy: SoftmaxPolicy, reference: SoftmaxPolicy) -> float:
    total = 0.0
    for prompt in range(domain.n_prompts):
        policy_distribution = policy.distribution(prompt)
        reference_distribution = reference.distribution(prompt)
        divergence = 0.0
        for probability, reference_probability in zip(
            policy_distribution, reference_distribution, strict=True
        ):
            if probability > 0.0:
                if reference_probability <= 0.0:
                    # The policy puts mass where the reference puts none, so the
                    # KL divergence is infinite; avoid dividing by zero.
                    divergence = math.inf
                    break
                divergence += probability * math.log(probability / reference_probability)
        total += divergence
    return total / domain.n_prompts


def evaluate(
    domain: Domain,
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    model: RewardModel,
    held_out: list[Preference],
) -> RLHFReport:
    """Produce a full RLHF report for the reward model and the policy."""

    return RLHFReport(
        reward_model_accuracy=reward_model_accuracy(domain, model, held_out),
        reward_model_correlation=reward_model_correlation(domain, model),
        reference_reward=expected_reward(domain, reference),
        policy_reward=expected_reward(domain, policy),
        optimal_reward=optimal_reward(domain),
        reward_model_ceiling=reward_model_ceiling(domain, model),
        mean_kl_to_reference=mean_kl(domain, policy, reference),
    )
