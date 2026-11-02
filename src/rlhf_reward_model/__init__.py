"""RLHF Reward Model - a Bradley-Terry reward model, PPO, and DPO from scratch."""

from rlhf_reward_model.comparison import (
    MethodComparison,
    MethodComparisonConfig,
    MethodResult,
    SampleEfficiencyRow,
    compare_ppo_dpo,
)
from rlhf_reward_model.dpo import DPOConfig, dpo_gradient, dpo_loss, train_dpo
from rlhf_reward_model.evaluation import RLHFReport, evaluate
from rlhf_reward_model.overoptimization import (
    OveroptimizationRow,
    OveroptimizationStudy,
    overoptimization_study,
)
from rlhf_reward_model.pipeline import PipelineConfig, PipelineResult, run_pipeline
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModel, RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone

__version__ = "0.1.0"

__all__ = [
    "DPOConfig",
    "MethodComparison",
    "MethodComparisonConfig",
    "MethodResult",
    "OveroptimizationRow",
    "OveroptimizationStudy",
    "PPOConfig",
    "PipelineConfig",
    "PipelineResult",
    "RLHFReport",
    "RewardModel",
    "RewardModelConfig",
    "SFTConfig",
    "SampleEfficiencyRow",
    "SoftmaxPolicy",
    "behavior_clone",
    "compare_ppo_dpo",
    "dpo_gradient",
    "dpo_loss",
    "evaluate",
    "overoptimization_study",
    "run_pipeline",
    "train_dpo",
    "train_ppo",
    "train_reward_model",
]
