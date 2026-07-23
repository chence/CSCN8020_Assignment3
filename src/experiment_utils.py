from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch

from g1_rl import G1ElbowTargetEnv


BENCHMARK_GOALS = (-0.8, -0.4, 0.4, 0.8)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    policy: Callable[[np.ndarray, dict[str, Any]], int],
    seed: int,
    episodes_per_goal: int = 5,
    render_mode: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_index = 0
    for goal in BENCHMARK_GOALS:
        env = G1ElbowTargetEnv(
            render_mode=render_mode,
            goal_angle=goal,
            goal_range=(-0.8, 0.8),
        )
        try:
            for goal_episode in range(episodes_per_goal):
                episode_seed = seed + episode_index
                observation, info = env.reset(seed=episode_seed)
                total_reward = 0.0
                action_counts = [0, 0, 0]
                while True:
                    action = int(policy(observation, info))
                    action_counts[action] += 1
                    observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    total_reward += reward
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "episode": episode_index + 1,
                        "goal_episode": goal_episode + 1,
                        "seed": episode_seed,
                        "goal": goal,
                        "success": int(bool(info["is_success"])),
                        "cumulative_reward": total_reward,
                        "episode_length": int(info["episode_step"]),
                        "final_absolute_error": float(info["absolute_error"]),
                        "decrease_actions": action_counts[0],
                        "hold_actions": action_counts[1],
                        "increase_actions": action_counts[2],
                    }
                )
                episode_index += 1
        finally:
            env.close()
    return rows


def summarize_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label, goal in [
        (f"{goal:+.1f}", goal) for goal in BENCHMARK_GOALS
    ] + [("overall", None)]:
        selected = rows if goal is None else [r for r in rows if r["goal"] == goal]
        successes = sum(int(r["success"]) for r in selected)
        summaries.append(
            {
                "goal": label,
                "episodes": len(selected),
                "successes": successes,
                "success_rate": successes / len(selected),
                "mean_reward": float(
                    np.mean([r["cumulative_reward"] for r in selected])
                ),
                "mean_episode_length": float(
                    np.mean([r["episode_length"] for r in selected])
                ),
                "mean_final_absolute_error": float(
                    np.mean([r["final_absolute_error"] for r in selected])
                ),
            }
        )
    return summaries
