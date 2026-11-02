"""Tests for Direct Preference Optimization."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rlhf_reward_model.domain import Domain, Preference, demonstrations, preferences
from rlhf_reward_model.dpo import (
    DPOConfig,
    dpo_gradient,
    dpo_loss,
    implicit_margin,
    train_dpo,
)
from rlhf_reward_model.evaluation import expected_reward, optimal_reward
from rlhf_reward_model.policy import SoftmaxPolicy
from rlhf_reward_model.sft import SFTConfig, behavior_clone


@dataclass(frozen=True)
class _Setup:
    domain: Domain
    reference: SoftmaxPolicy
    data: list[Preference]


@pytest.fixture(scope="module")
def setup() -> _Setup:
    domain = Domain.build(seed=0)
    reference = behavior_clone(
        domain, demonstrations(domain, 300, temperature=1.0, seed=1), SFTConfig(seed=2)
    )
    data = preferences(domain, 1500, seed=3)
    return _Setup(domain=domain, reference=reference, data=data)


def test_train_dpo_improves_true_reward(setup: _Setup) -> None:
    policy = train_dpo(setup.domain, setup.reference, setup.data, DPOConfig(iterations=250, seed=6))
    assert expected_reward(setup.domain, policy) > expected_reward(setup.domain, setup.reference)
    assert expected_reward(setup.domain, policy) <= optimal_reward(setup.domain) + 1e-9


def test_dpo_loss_decreases_with_training(setup: _Setup) -> None:
    before = dpo_loss(setup.domain, setup.reference, setup.reference, setup.data, beta=0.1)
    policy = train_dpo(
        setup.domain, setup.reference, setup.data, DPOConfig(iterations=250, beta=0.1, seed=6)
    )
    after = dpo_loss(setup.domain, policy, setup.reference, setup.data, beta=0.1)
    # At the reference the implicit margins are zero, so the loss is -log 0.5 = ln 2.
    assert before == pytest.approx(0.6931, abs=1e-3)
    assert after < before


def test_dpo_increases_chosen_margin_on_average(setup: _Setup) -> None:
    policy = train_dpo(
        setup.domain, setup.reference, setup.data, DPOConfig(iterations=250, beta=0.1, seed=6)
    )
    margins = [implicit_margin(policy, setup.reference, pref, 0.1) for pref in setup.data]
    assert sum(margins) / len(margins) > 0.0


def test_dpo_gradient_vanishes_at_zero_data(setup: _Setup) -> None:
    gradient = dpo_gradient(setup.domain, setup.reference, setup.reference, [], beta=0.1)
    assert gradient == [0.0] * setup.domain.dim
    assert dpo_loss(setup.domain, setup.reference, setup.reference, [], beta=0.1) == 0.0


def test_dpo_is_deterministic(setup: _Setup) -> None:
    first = train_dpo(setup.domain, setup.reference, setup.data, DPOConfig(iterations=80, seed=6))
    second = train_dpo(setup.domain, setup.reference, setup.data, DPOConfig(iterations=80, seed=6))
    assert first.weight == second.weight
