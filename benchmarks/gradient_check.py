"""Independent validation of the DPO gradient against NumPy.

``train_dpo`` ascends a hand-derived gradient of the DPO objective
``mean log sigmoid(beta * [ (logit(y_w) - logit(y_l)) - (logit_ref(y_w) - logit_ref(y_l)) ])``.
This module recomputes the same objective with NumPy and differentiates it by central finite
differences - an implementation sharing no code with the analytic gradient - and compares the
two. If the hand-derived gradient is correct they must agree to numerical precision.

NumPy is an optional ``validation`` dependency, so the accompanying test is skipped when it is
not installed; the core library does not require it. Run with:

    uv sync --group validation && uv run python benchmarks/gradient_check.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlhf_reward_model.domain import Domain, preferences
from rlhf_reward_model.dpo import dpo_gradient
from rlhf_reward_model.policy import SoftmaxPolicy


@dataclass(frozen=True)
class GradientCheck:
    """The outcome of comparing analytic and numerical DPO gradients."""

    max_abs_error: float
    analytic_norm: float
    numerical_norm: float


def _numpy_loss(
    weight: np.ndarray,
    reference_weight: np.ndarray,
    features: np.ndarray,
    data: list[tuple[int, int, int]],
    beta: float,
) -> float:
    """The mean DPO loss, recomputed independently with NumPy."""
    total = 0.0
    for prompt, chosen, rejected in data:
        phi_w = features[prompt, chosen]
        phi_l = features[prompt, rejected]
        policy_margin = float(weight @ phi_w - weight @ phi_l)
        reference_margin = float(reference_weight @ phi_w - reference_weight @ phi_l)
        h = beta * (policy_margin - reference_margin)
        total += float(np.logaddexp(0.0, -h))  # -log sigmoid(h) = softplus(-h)
    return total / len(data)


def check(*, beta: float = 0.1, seed: int = 0, epsilon: float = 1e-6) -> GradientCheck:
    """Compare the analytic DPO gradient against a NumPy finite-difference gradient."""
    domain = Domain.build(seed=seed)
    data = preferences(domain, 200, seed=seed + 3)
    reference = SoftmaxPolicy(domain=domain)
    # A non-trivial point: nudge the policy weights away from the reference.
    policy = SoftmaxPolicy(domain=domain, weight=[0.15 * (i + 1) for i in range(domain.dim)])

    analytic = np.array(dpo_gradient(domain, policy, reference, data, beta=beta))

    features = np.array(domain.features)
    reference_weight = np.array(reference.weight)
    weight = np.array(policy.weight)
    triples = [(p.prompt, p.chosen, p.rejected) for p in data]

    numerical = np.zeros_like(weight)
    for i in range(weight.shape[0]):
        up = weight.copy()
        up[i] += epsilon
        down = weight.copy()
        down[i] -= epsilon
        loss_up = _numpy_loss(up, reference_weight, features, triples, beta)
        loss_down = _numpy_loss(down, reference_weight, features, triples, beta)
        numerical[i] = (loss_up - loss_down) / (2.0 * epsilon)

    # dpo_gradient is the gradient of the objective (log sigmoid), i.e. the negative of the
    # gradient of the loss, so the analytic gradient of the loss is -analytic.
    analytic_loss_gradient = -analytic
    max_abs_error = float(np.abs(analytic_loss_gradient - numerical).max())
    return GradientCheck(
        max_abs_error=max_abs_error,
        analytic_norm=float(np.linalg.norm(analytic)),
        numerical_norm=float(np.linalg.norm(numerical)),
    )


def main() -> None:
    result = check()
    print("DPO gradient check (analytic vs NumPy finite differences):")
    print(f"  max abs error : {result.max_abs_error:.3e}")
    print(f"  analytic norm : {result.analytic_norm:.6f}")
    print(f"  numerical norm: {result.numerical_norm:.6f}")
    status = "PASS" if result.max_abs_error < 1e-6 else "FAIL"
    print(f"  {status}: the hand-derived DPO gradient matches the independent NumPy reference.")


if __name__ == "__main__":
    main()
