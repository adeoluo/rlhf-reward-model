"""The over-optimization table in the README and the evaluation report is current."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rlhf_reward_model.overoptimization import OveroptimizationStudy, overoptimization_study

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
