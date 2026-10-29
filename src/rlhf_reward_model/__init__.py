"""RLHF Reward Model - a Bradley-Terry reward model and PPO from scratch."""

from rlhf_reward_model.evaluation import RLHFReport, evaluate
from rlhf_reward_model.pipeline import PipelineConfig, PipelineResult, run_pipeline
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModel, RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone

__version__ = "0.1.0"

__all__ = [
    "PPOConfig",
    "PipelineConfig",
    "PipelineResult",
    "RLHFReport",
    "RewardModel",
    "RewardModelConfig",
    "SFTConfig",
    "SoftmaxPolicy",
    "behavior_clone",
    "evaluate",
    "run_pipeline",
    "train_ppo",
    "train_reward_model",
]
