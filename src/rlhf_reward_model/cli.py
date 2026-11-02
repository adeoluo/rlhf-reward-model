"""Command line interface for the RLHF reward-model pipeline."""

from __future__ import annotations

import argparse

from rlhf_reward_model.comparison import MethodComparisonConfig, compare_ppo_dpo
from rlhf_reward_model.dpo import DPOConfig
from rlhf_reward_model.overoptimization import overoptimization_study
from rlhf_reward_model.pipeline import PipelineConfig, run_pipeline
from rlhf_reward_model.ppo import PPOConfig


def _cmd_evaluate(args: argparse.Namespace) -> None:
    result = run_pipeline(PipelineConfig(beta=args.beta, seed=args.seed))
    print("RLHF pipeline (supervised fine-tuning, reward model, PPO):")
    print(f"  {result.report.format()}")


def _cmd_compare(args: argparse.Namespace) -> None:
    result = run_pipeline(PipelineConfig(beta=args.beta, seed=args.seed))
    domain = result.domain
    print("Per-prompt greedy choice (reference -> PPO policy, with the gold-best response):")
    for prompt in range(domain.n_prompts):
        reference_choice = result.reference.greedy(prompt)
        policy_choice = result.policy.greedy(prompt)
        best = domain.best_response(prompt)
        marker = "match" if policy_choice == best else "miss"
        print(
            f"  prompt {prompt}: reference r{reference_choice} "
            f"-> policy r{policy_choice}  (gold-best r{best}, {marker})"
        )


def _cmd_methods(args: argparse.Namespace) -> None:
    sizes = tuple(s for s in (250, 500, 1000, 2000, 4000) if s <= args.preferences)
    if not sizes:
        sizes = (args.preferences,)
    config = MethodComparisonConfig(
        preferences=args.preferences,
        sample_sizes=sizes,
        ppo=PPOConfig(beta=0.05, iterations=args.ppo_iterations, seed=args.seed + 6),
        dpo=DPOConfig(
            beta=0.05,
            iterations=args.dpo_iterations,
            learning_rate=1.0,
            l2=1e-4,
            seed=args.seed + 6,
        ),
        seed=args.seed,
    )
    comparison = compare_ppo_dpo(config)
    print("PPO vs DPO on the same task (shared reference, preferences, optimizer):")
    print(f"  reference {comparison.reference_reward:.4f}, optimal {comparison.optimal_reward:.4f}")
    print("  method   true reward   gap closed   mean KL")
    for result in (comparison.ppo, comparison.dpo):
        print(
            f"  {result.name:<7}  {result.true_reward:>10.4f}   "
            f"{result.reward_gap_closed:>9.1%}   {result.mean_kl_to_reference:>6.2f}"
        )
    print("  sample efficiency (true reward vs. preference pairs):")
    print("  preferences    PPO       DPO")
    for row in comparison.sample_efficiency:
        print(
            f"  {row.preferences:>11}   {row.ppo_true_reward:>7.4f}   {row.dpo_true_reward:>7.4f}"
        )


def _cmd_overoptimize(args: argparse.Namespace) -> None:
    study = overoptimization_study(
        demonstrations_count=args.demonstrations,
        preferences_count=args.preferences,
        ppo_iterations=args.ppo_iterations,
        seed=args.seed,
    )
    print(
        f"Reward over-optimization with a misspecified reward model "
        f"({study.active_features} of {study.total_features} features):"
    )
    print("  beta   proxy reward   true reward   mean KL")
    for row in study.rows:
        print(
            f"  {row.beta:<5}  {row.proxy_reward:>11.4f}   "
            f"{row.true_reward:>10.4f}   {row.mean_kl:>6.2f}"
        )
    print(f"  true reward peaks at beta={study.best_beta} ({study.best_true_reward:.4f}).")


def _cmd_serve(args: argparse.Namespace) -> None:
    from rlhf_reward_model.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlhf-reward-model", description=__doc__)
    parser.add_argument("--beta", type=float, default=0.05, help="PPO KL-penalty strength")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="run the full pipeline and print the RLHF report")
    sub.add_parser("compare", help="show the greedy choice per prompt before and after PPO")

    methods = sub.add_parser("methods", help="compare PPO and DPO head to head")
    methods.add_argument("--preferences", type=int, default=4000, help="preference pairs")
    methods.add_argument("--ppo-iterations", type=int, default=60, dest="ppo_iterations")
    methods.add_argument("--dpo-iterations", type=int, default=1000, dest="dpo_iterations")

    overopt = sub.add_parser("overoptimize", help="trace the reward over-optimization curve")
    overopt.add_argument("--preferences", type=int, default=4000, help="preference pairs")
    overopt.add_argument("--demonstrations", type=int, default=600, help="SFT demonstrations")
    overopt.add_argument("--ppo-iterations", type=int, default=60, dest="ppo_iterations")

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {
        "evaluate": _cmd_evaluate,
        "compare": _cmd_compare,
        "methods": _cmd_methods,
        "overoptimize": _cmd_overoptimize,
        "serve": _cmd_serve,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
