"""Direct Preference Optimization (DPO) on the log-linear policy.

DPO is the reward-model-free alternative to the classical reward-model + PPO pipeline. Rather
than fitting an explicit reward and then optimizing it with reinforcement learning, DPO shows
that the RLHF objective has a closed-form optimal policy, and that fitting the policy directly
to preferences with a single classification loss reaches the same optimum. The loss for a
preference (chosen y_w over rejected y_l given prompt x) is

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                               - (log pi(y_l|x) - log pi_ref(y_l|x)) ] ),

the implicit reward being ``beta * log(pi / pi_ref)``. For the log-linear policy used here the
log-probability difference ``log pi(y_w|x) - log pi(y_l|x)`` equals the logit difference (the
normalizer cancels), so the whole gradient is exact and closed-form:

    dL/dw = -sigmoid(-h) * beta * (phi(x, y_w) - phi(x, y_l)),

with ``h`` the bracketed term above. The policy is initialized from the reference and optimized
by full-batch gradient ascent with an L2 penalty - the same numerical recipe as the reward
model, so PPO and DPO differ in *method*, not in optimizer quality, which is what makes the
head-to-head comparison fair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rlhf_reward_model.domain import Domain, Preference
from rlhf_reward_model.policy import SoftmaxPolicy


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class DPOConfig:
    """DPO hyperparameters (full-batch gradient ascent)."""

    iterations: int = 400
    learning_rate: float = 0.5
    beta: float = 0.1
    l2: float = 1e-3
    seed: int = 0


def _logit(policy: SoftmaxPolicy, prompt: int, response: int) -> float:
    phi = policy.domain.feature(prompt, response)
    return sum(w * x for w, x in zip(policy.weight, phi, strict=True))


def implicit_margin(
    policy: SoftmaxPolicy, reference: SoftmaxPolicy, preference: Preference, beta: float
) -> float:
    """The DPO logit ``h`` for one preference.

    Because the policy is log-linear, ``log pi(y|x)`` and the raw logit ``w . phi(x, y)``
    differ only by the per-prompt normalizer, which cancels in the chosen-minus-rejected
    difference. So ``h`` is ``beta`` times the policy's preference margin minus the
    reference's preference margin.
    """
    policy_margin = _logit(policy, preference.prompt, preference.chosen) - _logit(
        policy, preference.prompt, preference.rejected
    )
    reference_margin = _logit(reference, preference.prompt, preference.chosen) - _logit(
        reference, preference.prompt, preference.rejected
    )
    return beta * (policy_margin - reference_margin)


def dpo_loss(
    domain: Domain,
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    data: list[Preference],
    *,
    beta: float,
) -> float:
    """Mean DPO loss over ``data`` (lower is better)."""
    if not data:
        return 0.0
    total = 0.0
    for preference in data:
        h = implicit_margin(policy, reference, preference, beta)
        # -log sigmoid(h), numerically stable via softplus(-h).
        total += _softplus(-h)
    return total / len(data)


def dpo_gradient(
    domain: Domain,
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    data: list[Preference],
    *,
    beta: float,
) -> list[float]:
    """The mean gradient of the DPO objective w.r.t. the policy weights (no L2 term)."""
    gradient = [0.0] * domain.dim
    if not data:
        return gradient
    for preference in data:
        h = implicit_margin(policy, reference, preference, beta)
        scale = _sigmoid(-h) * beta
        phi_w = domain.feature(preference.prompt, preference.chosen)
        phi_l = domain.feature(preference.prompt, preference.rejected)
        for index in range(domain.dim):
            gradient[index] += scale * (phi_w[index] - phi_l[index])
    inverse = 1.0 / len(data)
    return [value * inverse for value in gradient]


def _softplus(value: float) -> float:
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def train_dpo(
    domain: Domain,
    reference: SoftmaxPolicy,
    data: list[Preference],
    config: DPOConfig,
) -> SoftmaxPolicy:
    """Optimize a copy of the reference on preferences with the DPO objective.

    The DPO objective is the logistic loss of the implicit reward margin and is concave in the
    weights, so full-batch gradient ascent converges stably. Each iteration averages the exact
    per-pair gradient and applies an L2 shrinkage term that keeps the policy near the reference.
    """
    policy = reference.copy()
    for _ in range(config.iterations):
        gradient = dpo_gradient(domain, policy, reference, data, beta=config.beta)
        for index in range(domain.dim):
            policy.weight[index] += config.learning_rate * (
                gradient[index] - config.l2 * policy.weight[index]
            )
    return policy
