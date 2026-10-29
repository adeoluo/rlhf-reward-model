"""Command line interface for the RLHF reward-model pipeline."""

from __future__ import annotations

import argparse

from rlhf_reward_model.pipeline import PipelineConfig, run_pipeline


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

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {"evaluate": _cmd_evaluate, "compare": _cmd_compare, "serve": _cmd_serve}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
