"""Tests for the RLHF CLI subcommands."""

from __future__ import annotations

import pytest

from rlhf_reward_model.cli import main


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_evaluate_and_compare(capsys: pytest.CaptureFixture[str]) -> None:
    main(["evaluate"])
    main(["compare"])
    out = capsys.readouterr().out
    assert "RLHF pipeline" in out
    assert "Per-prompt greedy choice" in out


def test_cli_methods_and_overoptimize(capsys: pytest.CaptureFixture[str]) -> None:
    main(["methods", "--preferences", "250", "--ppo-iterations", "10", "--dpo-iterations", "50"])
    main(
        [
            "overoptimize",
            "--preferences",
            "400",
            "--demonstrations",
            "100",
            "--ppo-iterations",
            "10",
        ]
    )
    out = capsys.readouterr().out
    assert "PPO vs DPO" in out
    assert "true reward peaks at beta=" in out
