"""Reward over-optimization (Goodhart's law) as a measured, first-class study.

A reward model is a *proxy* for the hidden true reward. Optimizing a proxy harder does not keep
improving the true objective: past a point the policy exploits the proxy's errors and the true
reward turns over. This module reproduces that curve with a deliberately **misspecified** reward
model (fit on only a subset of the features, so it cannot represent the true reward) and PPO run
at a decreasing KL penalty ``beta``. As ``beta`` falls the policy is freer to chase the proxy:
the proxy reward keeps climbing while the true reward rises, peaks, and then declines.

The study is deterministic and exposed as a library function with a returned table, so it is
unit-tested rather than living only in a benchmark script.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import expected_reward, mean_kl
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModel, RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


@dataclass(frozen=True)
class OveroptimizationRow:
    """One point on the over-optimization curve, at a given KL penalty ``beta``."""

    beta: float
    proxy_reward: float
    true_reward: float
    mean_kl: float


@dataclass(frozen=True)
class OveroptimizationStudy:
    """The over-optimization curve plus the true-reward peak it reveals."""

    rows: tuple[OveroptimizationRow, ...]
    active_features: int
    total_features: int

    @property
    def best_true_reward(self) -> float:
        return max(row.true_reward for row in self.rows)

    @property
    def best_beta(self) -> float:
        """The ``beta`` that maximizes the true reward (the sweet spot before over-optimizing)."""
        return max(self.rows, key=lambda row: row.true_reward).beta


def _expected_proxy_reward(domain: Domain, policy: SoftmaxPolicy, model: RewardModel) -> float:
    total = 0.0
    for prompt in range(domain.n_prompts):
        distribution = policy.distribution(prompt)
        total += sum(
            probability * model.reward(prompt, response)
            for response, probability in enumerate(distribution)
        )
    return total / domain.n_prompts


def overoptimization_study(
    *,
    betas: tuple[float, ...] = (0.2, 0.1, 0.05, 0.02, 0.01),
    active_features: int = 4,
    demonstrations_count: int = 600,
    preferences_count: int = 4000,
    ppo_iterations: int = 60,
    seed: int = 0,
) -> OveroptimizationStudy:
    """Run PPO against a misspecified reward model at decreasing KL penalty.

    The ``betas`` are expected in decreasing order (looser and looser KL budget). The returned
    rows trace the classic Goodhart curve: a monotonically rising proxy reward against a true
    reward that peaks and then falls. ``ppo_iterations`` controls the optimization effort; the
    over-optimization turn-over needs the default budget, but smoke tests can lower it.
    """
    domain = Domain.build(seed=seed)
    sft_data = demonstrations(domain, demonstrations_count, temperature=1.0, seed=seed + 1)
    reference = behavior_clone(domain, sft_data, SFTConfig(seed=seed + 2))
    train = preferences(domain, preferences_count, seed=seed + 3)
    model = train_reward_model(
        domain, train, RewardModelConfig(active_features=active_features, seed=seed + 5)
    )

    rows: list[OveroptimizationRow] = []
    for beta in betas:
        policy = train_ppo(
            domain, reference, model, PPOConfig(beta=beta, iterations=ppo_iterations, seed=seed + 6)
        )
        rows.append(
            OveroptimizationRow(
                beta=beta,
                proxy_reward=_expected_proxy_reward(domain, policy, model),
                true_reward=expected_reward(domain, policy),
                mean_kl=mean_kl(domain, policy, reference),
            )
        )
    return OveroptimizationStudy(
        rows=tuple(rows),
        active_features=active_features,
        total_features=domain.dim,
    )
