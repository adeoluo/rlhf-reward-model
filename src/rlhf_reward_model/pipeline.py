"""The end-to-end RLHF pipeline: SFT, reward model, PPO, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import RLHFReport, evaluate
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModel, RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level knobs for the data and the three training stages."""

    demonstrations: int = 600
    preferences: int = 4000
    held_out_preferences: int = 800
    sft_temperature: float = 1.0
    beta: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class PipelineResult:
    """The trained reference, reward model, and policy, with the report."""

    domain: Domain
    reference: SoftmaxPolicy
    reward_model: RewardModel
    policy: SoftmaxPolicy
    report: RLHFReport


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Run supervised fine-tuning, reward modeling, and PPO, then evaluate."""

    config = config or PipelineConfig()
    domain = Domain.build(seed=config.seed)
    sft_data = demonstrations(
        domain, config.demonstrations, temperature=config.sft_temperature, seed=config.seed + 1
    )
    reference = behavior_clone(domain, sft_data, SFTConfig(seed=config.seed + 2))
    train_preferences = preferences(domain, config.preferences, seed=config.seed + 3)
    held_out = preferences(domain, config.held_out_preferences, seed=config.seed + 4)
    reward_model = train_reward_model(
        domain, train_preferences, RewardModelConfig(seed=config.seed + 5)
    )
    policy = train_ppo(
        domain, reference, reward_model, PPOConfig(beta=config.beta, seed=config.seed + 6)
    )
    report = evaluate(domain, policy, reference, reward_model, held_out)
    return PipelineResult(
        domain=domain,
        reference=reference,
        reward_model=reward_model,
        policy=policy,
        report=report,
    )
