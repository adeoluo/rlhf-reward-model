"""Tests for the PPO-vs-DPO comparison and the over-optimization study."""

from __future__ import annotations

import pytest

from rlhf_reward_model.comparison import (
    MethodComparison,
    MethodComparisonConfig,
    compare_ppo_dpo,
)
from rlhf_reward_model.dpo import DPOConfig
from rlhf_reward_model.overoptimization import overoptimization_study
from rlhf_reward_model.ppo import PPOConfig
from rlhf_reward_model.reward_model import RewardModelConfig


@pytest.fixture(scope="module")
def comparison() -> MethodComparison:
    config = MethodComparisonConfig(
        demonstrations=300,
        preferences=800,
        sample_sizes=(200, 800),
        reward_model=RewardModelConfig(iterations=150, seed=5),
        ppo=PPOConfig(iterations=25, seed=6),
        dpo=DPOConfig(beta=0.05, iterations=250, learning_rate=1.0, l2=1e-4, seed=6),
    )
    return compare_ppo_dpo(config)


def test_compare_ppo_dpo_both_improve_over_reference(comparison: MethodComparison) -> None:
    assert comparison.optimal_reward > comparison.reference_reward
    assert comparison.ppo.true_reward > comparison.reference_reward
    assert comparison.dpo.true_reward > comparison.reference_reward
    assert comparison.ppo.true_reward <= comparison.optimal_reward + 1e-9
    assert comparison.dpo.true_reward <= comparison.optimal_reward + 1e-9


def test_sample_efficiency_table_matches_sizes(comparison: MethodComparison) -> None:
    assert [row.preferences for row in comparison.sample_efficiency] == [200, 800]
    for row in comparison.sample_efficiency:
        assert row.ppo_true_reward > comparison.reference_reward - 0.5
        assert row.dpo_true_reward > comparison.reference_reward - 0.5


def test_method_comparison_formats(comparison: MethodComparison) -> None:
    text = comparison.format()
    assert "PPO" in text
    assert "DPO" in text


@pytest.mark.slow
def test_overoptimization_curve_turns_over() -> None:
    # At full data budget the misspecified reward's true reward peaks in the interior and
    # then declines as the KL penalty falls - the Goodhart over-optimization signature.
    # This needs the full training budget, so it is marked slow and excluded from the
    # default CI run; the benchmark reproduces the full curve.
    study = overoptimization_study()
    assert study.active_features < study.total_features
    betas = [row.beta for row in study.rows]
    assert study.best_beta not in (betas[0], betas[-1])
    assert study.best_true_reward > study.rows[-1].true_reward
    # The proxy reward rises overall as the KL penalty is loosened.
    assert study.rows[-1].proxy_reward > study.rows[0].proxy_reward


def test_overoptimization_study_structure() -> None:
    study = overoptimization_study(
        demonstrations_count=200,
        preferences_count=500,
        betas=(0.2, 0.05),
        ppo_iterations=8,
        seed=0,
    )
    assert len(study.rows) == 2
    for row in study.rows:
        assert row.mean_kl >= 0.0
