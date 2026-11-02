"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

from rlhf_reward_model.cli import main


def test_cli_methods(capsys: pytest.CaptureFixture[str]) -> None:
    main(["methods", "--preferences", "300", "--ppo-iterations", "5", "--dpo-iterations", "50"])
    output = capsys.readouterr().out
    assert "PPO vs DPO" in output
    assert "sample efficiency" in output
    assert "ppo" in output
    assert "dpo" in output


def test_cli_overoptimize(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "overoptimize",
            "--preferences",
            "400",
            "--demonstrations",
            "150",
            "--ppo-iterations",
            "8",
        ]
    )
    output = capsys.readouterr().out
    assert "over-optimization" in output
    assert "proxy reward" in output
    assert "beta" in output


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
