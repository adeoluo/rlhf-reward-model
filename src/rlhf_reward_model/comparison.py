"""A head-to-head comparison of PPO and DPO on the same synthetic preference task.

PPO and DPO solve the same alignment problem by different routes. PPO fits an explicit
Bradley-Terry reward model and then optimizes it with a clipped policy gradient under a KL
penalty; DPO skips the reward model and fits the policy directly to the preferences. Both are
given the *same* reference, the *same* preferences, and the *same* full-batch optimizer, so the
comparison isolates the method rather than incidental tuning.

Two questions are reported:

- **Final quality.** The true reward each method reaches (and the fraction of the
  reference-to-optimal gap it closes), plus the mean KL to the reference each spends to get
  there.
- **Sample efficiency.** The true reward each method reaches as the number of preference pairs
  grows, which exposes how much data each route needs.

Everything is deterministic and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlhf_reward_model.domain import Domain, Preference, demonstrations, preferences
from rlhf_reward_model.dpo import DPOConfig, train_dpo
from rlhf_reward_model.evaluation import expected_reward, mean_kl, optimal_reward
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


@dataclass(frozen=True)
class MethodResult:
    """How far one method moved the policy, and at what KL cost."""

    name: str
    true_reward: float
    mean_kl_to_reference: float
    reward_gap_closed: float


@dataclass(frozen=True)
class SampleEfficiencyRow:
    """True reward reached by each method at a given preference-set size."""

    preferences: int
    ppo_true_reward: float
    dpo_true_reward: float


@dataclass(frozen=True)
class MethodComparison:
    """The full PPO-vs-DPO comparison on a shared task."""

    reference_reward: float
    optimal_reward: float
    ppo: MethodResult
    dpo: MethodResult
    sample_efficiency: tuple[SampleEfficiencyRow, ...]

    def format(self) -> str:
        """Render a short, human-readable summary."""
        return (
            f"reference {self.reference_reward:.4f}, optimal {self.optimal_reward:.4f}  |  "
            f"PPO true {self.ppo.true_reward:.4f} (gap {self.ppo.reward_gap_closed:.1%}, "
            f"KL {self.ppo.mean_kl_to_reference:.2f})  |  "
            f"DPO true {self.dpo.true_reward:.4f} (gap {self.dpo.reward_gap_closed:.1%}, "
            f"KL {self.dpo.mean_kl_to_reference:.2f})"
        )


@dataclass(frozen=True)
class MethodComparisonConfig:
    """Knobs for the comparison; the sub-stage configs carry their own hyperparameters."""

    demonstrations: int = 600
    preferences: int = 4000
    sample_sizes: tuple[int, ...] = (250, 500, 1000, 2000, 4000)
    sft: SFTConfig = field(default_factory=lambda: SFTConfig(seed=2))
    reward_model: RewardModelConfig = field(default_factory=lambda: RewardModelConfig(seed=5))
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(beta=0.05, seed=6))
    dpo: DPOConfig = field(
        default_factory=lambda: DPOConfig(
            beta=0.05, iterations=1000, learning_rate=1.0, l2=1e-4, seed=6
        )
    )
    seed: int = 0


def _result(
    name: str, domain: Domain, policy: SoftmaxPolicy, reference: SoftmaxPolicy
) -> MethodResult:
    reference_reward = expected_reward(domain, reference)
    optimal = optimal_reward(domain)
    true_reward = expected_reward(domain, policy)
    span = optimal - reference_reward
    gap = (true_reward - reference_reward) / span if span > 0.0 else 0.0
    return MethodResult(
        name=name,
        true_reward=true_reward,
        mean_kl_to_reference=mean_kl(domain, policy, reference),
        reward_gap_closed=gap,
    )


def _ppo_policy(
    domain: Domain,
    reference: SoftmaxPolicy,
    data: list[Preference],
    config: MethodComparisonConfig,
) -> SoftmaxPolicy:
    reward_model = train_reward_model(domain, data, config.reward_model)
    return train_ppo(domain, reference, reward_model, config.ppo)


def compare_ppo_dpo(config: MethodComparisonConfig | None = None) -> MethodComparison:
    """Train PPO and DPO on the same task and report quality and sample efficiency."""
    config = config or MethodComparisonConfig()
    domain = Domain.build(seed=config.seed)
    sft_data = demonstrations(domain, config.demonstrations, temperature=1.0, seed=config.seed + 1)
    reference = behavior_clone(domain, sft_data, config.sft)
    full_preferences = preferences(domain, config.preferences, seed=config.seed + 3)

    ppo_policy = _ppo_policy(domain, reference, full_preferences, config)
    dpo_policy = train_dpo(domain, reference, full_preferences, config.dpo)

    rows: list[SampleEfficiencyRow] = []
    for size in config.sample_sizes:
        subset = full_preferences[:size]
        ppo_subset = _ppo_policy(domain, reference, subset, config)
        dpo_subset = train_dpo(domain, reference, subset, config.dpo)
        rows.append(
            SampleEfficiencyRow(
                preferences=size,
                ppo_true_reward=expected_reward(domain, ppo_subset),
                dpo_true_reward=expected_reward(domain, dpo_subset),
            )
        )

    return MethodComparison(
        reference_reward=expected_reward(domain, reference),
        optimal_reward=optimal_reward(domain),
        ppo=_result("ppo", domain, ppo_policy, reference),
        dpo=_result("dpo", domain, dpo_policy, reference),
        sample_efficiency=tuple(rows),
    )
