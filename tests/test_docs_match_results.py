"""The over-optimization table in the README and the evaluation report is current."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rlhf_reward_model.comparison import compare_ppo_dpo
from rlhf_reward_model.overoptimization import OveroptimizationStudy, overoptimization_study
from rlhf_reward_model.pipeline import PipelineConfig, run_pipeline

ROOT = Path(__file__).resolve().parent.parent
ROW = re.compile(r"^\| (0\.\d+) \| ([\d.]+) \| \**([\d.]+)\** \| ([\d.]+) \|$", re.M)


@pytest.fixture(scope="module")
def study() -> OveroptimizationStudy:
    return overoptimization_study()


@pytest.mark.parametrize("path", ["README.md", "docs/EVALUATION_REPORT.md"])
def test_overoptimization_table_matches_a_fresh_run(
    path: str, study: OveroptimizationStudy
) -> None:
    quoted = {
        float(m[1]): tuple(float(x) for x in m.groups()[1:])
        for m in ROW.finditer((ROOT / path).read_text())
    }
    assert set(quoted) == {row.beta for row in study.rows}, path
    fresh = [
        (row.beta, round(row.proxy_reward, 3), round(row.true_reward, 3), round(row.mean_kl, 2))
        for row in study.rows
    ]
    for row in study.rows:
        proxy, true, kl = quoted[row.beta]
        assert proxy == pytest.approx(row.proxy_reward, abs=2e-3), (path, fresh)
        assert true == pytest.approx(row.true_reward, abs=2e-3), (path, fresh)
        assert kl == pytest.approx(row.mean_kl, abs=2e-2), (path, fresh)


def test_headline_table_matches_a_fresh_run() -> None:
    report = run_pipeline(PipelineConfig()).report
    fresh = [
        f"correlation (Pearson)** | **{report.reward_model_correlation:.3f}** |",
        f"| reward model held-out ranking accuracy | {report.reward_model_accuracy:.3f} |",
        f"(reference to optimal)** | **{report.reward_gap_closed:.1%}** |",
        f"| policy true reward (reference to PPO) | {report.reference_reward:.3f} to "
        f"{report.policy_reward:.3f} (optimal {report.optimal_reward:.3f}) |",
        f"| mean KL to reference | {report.mean_kl_to_reference:.3f} |",
    ]
    readme = (ROOT / "README.md").read_text()
    assert all(line in readme for line in fresh), fresh


def test_method_comparison_matches_a_fresh_run() -> None:
    result = compare_ppo_dpo()
    fresh = [
        f"**{result.ppo.true_reward:.4f}** | **{result.ppo.reward_gap_closed:.1%}** | "
        f"{result.ppo.mean_kl_to_reference:.2f} |",
        f"{result.dpo.true_reward:.4f} | {result.dpo.reward_gap_closed:.1%} | "
        f"**{result.dpo.mean_kl_to_reference:.2f}** |",
    ]
    readme = (ROOT / "README.md").read_text()
    assert all(line in readme for line in fresh), fresh
