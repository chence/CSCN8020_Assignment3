from __future__ import annotations

import argparse
from pathlib import Path

from dqn import DQNAgent
from experiment_utils import run_evaluation, summarize_evaluation, write_csv
from test_g1_elbow_env import choose_rule_based_action


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN and rule-based policies.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--episodes-per-goal", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-rule-based", action="store_true")
    return parser.parse_args()


def print_summary(name: str, summary: list[dict[str, object]]) -> None:
    print(f"\n{name}")
    for row in summary:
        print(
            f"goal={row['goal']:>7} successes={row['successes']}/{row['episodes']} "
            f"rate={float(row['success_rate']):.1%} "
            f"mean_reward={float(row['mean_reward']):+.3f}"
        )


def main() -> None:
    args = parse_arguments()
    agent, _ = DQNAgent.load(args.checkpoint, args.device)
    agent.online_network.eval()
    dqn_rows = run_evaluation(
        lambda observation, _info: agent.select_action(observation, epsilon=0.0),
        seed=args.seed,
        episodes_per_goal=args.episodes_per_goal,
    )
    dqn_summary = summarize_evaluation(dqn_rows)
    write_csv(args.output_dir / "dqn_evaluation_episodes.csv", dqn_rows)
    write_csv(args.output_dir / "dqn_evaluation_summary.csv", dqn_summary)
    print_summary("DQN (epsilon=0.0)", dqn_summary)

    if not args.skip_rule_based:
        rule_rows = run_evaluation(
            lambda observation, info: choose_rule_based_action(
                observation,
                float(info["controller_target"]),
                0.08,
            ),
            seed=args.seed,
            episodes_per_goal=args.episodes_per_goal,
        )
        rule_summary = summarize_evaluation(rule_rows)
        write_csv(args.output_dir / "rule_based_episodes.csv", rule_rows)
        write_csv(args.output_dir / "rule_based_summary.csv", rule_summary)
        print_summary("Rule-based baseline", rule_summary)


if __name__ == "__main__":
    main()
