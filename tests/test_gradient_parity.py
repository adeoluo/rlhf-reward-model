"""Independent parity check for the DPO gradient.

``train_dpo`` ascends a hand-derived gradient. This test recomputes the same DPO loss with
NumPy and differentiates it by finite differences - an independent implementation - and
asserts the two gradients agree. NumPy is an optional ``validation`` dependency, so this test
is skipped when it is not installed; the core library does not require it.

Run with: uv sync --group validation && uv run pytest tests/test_gradient_parity.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from benchmarks.gradient_check import check


def test_analytic_dpo_gradient_matches_numpy_finite_differences() -> None:
    result = check()
    assert result.max_abs_error < 1e-6
    # Sanity: the gradient is non-trivial, so the match is meaningful, not two zeros.
    assert result.analytic_norm > 1e-3
    assert result.numerical_norm > 1e-3
