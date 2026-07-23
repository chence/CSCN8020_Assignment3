from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = np.mean(values[start : index + 1])
    return result


def plot_training(config_dir: Path) -> None:
    rows = read_csv(config_dir / "training_metrics.csv")
    episodes = np.array([int(r["episode"]) for r in rows])
    rewards = np.array([float(r["cumulative_reward"]) for r in rows])
    successes = np.array([float(r["success"]) for r in rows])
    epsilons = np.array([float(r["epsilon"]) for r in rows])
    losses = np.array([float(r["mean_loss"]) for r in rows])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(episodes, rewards, alpha=0.3, label="Raw reward")
    axes[0, 0].plot(episodes, rolling(rewards, 20), label="20-episode mean")
    axes[0, 0].set_title("Training reward")
    axes[0, 0].legend()
    axes[0, 1].plot(episodes, rolling(successes, 50))
    axes[0, 1].set_title("50-episode success rate")
    axes[0, 1].set_ylim(0, 1.05)
    axes[1, 0].plot(episodes, epsilons)
    axes[1, 0].set_title("Epsilon")
    valid = np.isfinite(losses)
    axes[1, 1].plot(episodes[valid], losses[valid])
    axes[1, 1].set_title("Mean Huber loss")
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(config_dir / "training_plots.png", dpi=180)
    plt.close(figure)


def plot_evaluation(config_dir: Path) -> None:
    path = config_dir / "dqn_evaluation_summary.csv"
    if not path.exists():
        return
    rows = [r for r in read_csv(path) if r["goal"] != "overall"]
    goals = [r["goal"] for r in rows]
    rates = [float(r["success_rate"]) for r in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(goals, rates)
    axis.axhline(0.8, color="red", linestyle="--", label="80% target")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Goal angle (rad)")
    axis.set_ylabel("Success rate")
    axis.set_title("Greedy evaluation success by target")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config_dir / "evaluation_success_by_goal.png", dpi=180)
    plt.close(figure)


def plot_comparison(config_dirs: list[Path]) -> None:
    if len(config_dirs) < 2:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for config_dir in config_dirs:
        rows = read_csv(config_dir / "training_metrics.csv")
        episodes = np.array([int(r["episode"]) for r in rows])
        rewards = np.array([float(r["cumulative_reward"]) for r in rows])
        successes = np.array([float(r["success"]) for r in rows])
        label = config_dir.name
        axes[0].plot(episodes, rolling(rewards, 20), label=label)
        axes[1].plot(episodes, rolling(successes, 50), label=label)
    axes[0].set_title("20-episode mean reward")
    axes[1].set_title("50-episode success rate")
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output_path = config_dirs[0].parent / "epsilon_decay_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DQN experiment metrics.")
    parser.add_argument("config_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for config_dir in args.config_dirs:
        plot_training(config_dir)
        plot_evaluation(config_dir)
        print(f"Wrote plots under {config_dir}")
    plot_comparison(args.config_dirs)


if __name__ == "__main__":
    main()
