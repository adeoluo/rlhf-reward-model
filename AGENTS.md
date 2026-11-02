# AGENTS.md

Working notes for coding agents and contributors in **RLHF Reward Model**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Classical RLHF built from scratch: a Bradley-Terry reward model with PPO, a from-scratch DPO implementation and a head-to-head PPO-vs-DPO comparison, and an explicit reward over-optimization study.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/rlhf_reward_model/__init__.py`: RLHF Reward Model - a Bradley-Terry reward model, PPO, and DPO from scratch.
- `src/rlhf_reward_model/cli.py`: Command line interface for the RLHF reward-model pipeline.
- `src/rlhf_reward_model/comparison.py`: A head-to-head comparison of PPO and DPO on the same synthetic preference task.
- `src/rlhf_reward_model/domain.py`: The alignment task: prompts, candidate responses, and a hidden gold reward.
- `src/rlhf_reward_model/dpo.py`: Direct Preference Optimization (DPO) on the log-linear policy.
- `src/rlhf_reward_model/evaluation.py`: Evaluation for the RLHF pipeline: reward model fidelity and policy improvement.
- `src/rlhf_reward_model/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/rlhf_reward_model/overoptimization.py`: Reward over-optimization (Goodhart's law) as a measured, first-class study.
- `src/rlhf_reward_model/pipeline.py`: The end-to-end RLHF pipeline: SFT, reward model, PPO, and evaluation.
- `src/rlhf_reward_model/policy.py`: A log-linear softmax policy over candidate responses.
- `src/rlhf_reward_model/ppo.py`: PPO-style policy optimization against a learned reward.
- `src/rlhf_reward_model/reward_model.py`: A Bradley-Terry reward model fit from pairwise preferences.
- `src/rlhf_reward_model/service.py`: A standard-library HTTP service exposing the RLHF reward model and policy.
- `src/rlhf_reward_model/sft.py`: Supervised fine-tuning by behavior cloning.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
