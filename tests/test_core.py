"""Tests for the reward model, PPO, and evaluation."""

from __future__ import annotations

from rlhf_reward_model.domain import Domain, demonstrations, preferences
from rlhf_reward_model.evaluation import (
    evaluate,
    expected_reward,
    optimal_reward,
    reward_model_accuracy,
    reward_model_correlation,
)
from rlhf_reward_model.ppo import PPOConfig, train_ppo
from rlhf_reward_model.reward_model import RewardModelConfig, train_reward_model
from rlhf_reward_model.sft import SFTConfig, behavior_clone


def test_reward_model_recovers_true_reward() -> None:
    domain = Domain.build(seed=0)
    train = preferences(domain, 4000, seed=3)
    held_out = preferences(domain, 800, seed=4)
    model = train_reward_model(domain, train, RewardModelConfig(iterations=300, seed=5))
    assert reward_model_correlation(domain, model) > 0.95
    assert reward_model_accuracy(domain, model, held_out) > 0.9


def test_misspecified_reward_model_is_worse() -> None:
    domain = Domain.build(seed=0)
    train = preferences(domain, 4000, seed=3)
    full = train_reward_model(domain, train, RewardModelConfig(iterations=300, seed=5))
    partial = train_reward_model(
        domain, train, RewardModelConfig(iterations=300, active_features=4, seed=5)
    )
    assert reward_model_correlation(domain, partial) < reward_model_correlation(domain, full)


def test_ppo_improves_policy_true_reward() -> None:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 600, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 4000, seed=3)
    model = train_reward_model(domain, train, RewardModelConfig(iterations=300, seed=5))
    policy = train_ppo(domain, reference, model, PPOConfig(iterations=40, seed=6))

    assert expected_reward(domain, policy) > expected_reward(domain, reference)
    assert expected_reward(domain, policy) <= optimal_reward(domain) + 1e-9


def test_evaluate_report_is_consistent() -> None:
    domain = Domain.build(seed=0)
    sft = demonstrations(domain, 400, temperature=1.0, seed=1)
    reference = behavior_clone(domain, sft, SFTConfig(seed=2))
    train = preferences(domain, 3000, seed=3)
    held_out = preferences(domain, 400, seed=4)
    model = train_reward_model(domain, train, RewardModelConfig(iterations=250, seed=5))
    policy = train_ppo(domain, reference, model, PPOConfig(iterations=40, seed=6))
    report = evaluate(domain, policy, reference, model, held_out)

    assert 0.0 <= report.reward_model_accuracy <= 1.0
    assert report.mean_kl_to_reference >= 0.0
    assert report.reference_reward <= report.optimal_reward
    assert report.reward_model_ceiling <= report.optimal_reward + 1e-9
