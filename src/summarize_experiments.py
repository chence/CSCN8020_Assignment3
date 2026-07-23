from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from experiment_utils import write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the assignment comparison tables.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    experiment_rows: list[dict[str, object]] = []
    for name in ("config_a", "config_b"):
        directory = args.results_root / name
        training = read_csv(directory / "training_metrics.csv")
        evaluation = read_csv(directory / "dqn_evaluation_summary.csv")[-1]
        metadata = json.loads((directory / "training_summary.json").read_text())
        experiment_rows.append(
            {
                "configuration": name,
                "epsilon_decay": metadata["epsilon_decay"],
                "training_episodes": len(training),
                "training_seconds": metadata["training_seconds"],
                "final_epsilon": metadata["epsilon_final"],
                "final_20_mean_training_reward": statistics.mean(
                    float(row["cumulative_reward"]) for row in training[-20:]
                ),
                "final_50_training_success_rate": statistics.mean(
                    int(row["success"]) for row in training[-50:]
                ),
                "evaluation_success_rate": float(evaluation["success_rate"]),
                "mean_evaluation_reward": float(evaluation["mean_reward"]),
                "mean_evaluation_length": float(evaluation["mean_episode_length"]),
                "mean_final_absolute_error": float(
                    evaluation["mean_final_absolute_error"]
                ),
            }
        )
    write_csv(args.results_root / "epsilon_decay_comparison.csv", experiment_rows)

    selected = read_csv(args.results_root / "config_a" / "dqn_evaluation_summary.csv")[-1]
    baseline = read_csv(args.results_root / "config_a" / "rule_based_summary.csv")[-1]
    policy_rows = []
    for policy, row in (("rule_based", baseline), ("selected_dqn_config_a", selected)):
        policy_rows.append(
            {
                "policy": policy,
                "successes": row["successes"],
                "episodes": row["episodes"],
                "success_rate": row["success_rate"],
                "mean_cumulative_reward": row["mean_reward"],
                "mean_episode_length": row["mean_episode_length"],
                "mean_final_absolute_error": row["mean_final_absolute_error"],
            }
        )
    write_csv(args.results_root / "rule_based_vs_selected_dqn.csv", policy_rows)
    print(f"Wrote comparison tables under {args.results_root}")


if __name__ == "__main__":
    main()
