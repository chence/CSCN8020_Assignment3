from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from actor_critic import PPOAgent, PPOConfig, generalized_advantage_estimate
from inspire_reach_env import InspireReachEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the G1 Inspire reach task with PPO Actor-Critic."
    )
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", type=Path, default=Path("models/inspire_actor_critic.pt"))
    parser.add_argument("--results", type=Path, default=Path("results/inspire_actor_critic"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(
    agent: PPOAgent, env: InspireReachEnv, episodes: int, seed: int
) -> tuple[float, float]:
    successes = 0
    minimum_distances: list[float] = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        minimum_distance = float(info["distance"])
        while True:
            action, _, _ = agent.select_action(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            minimum_distance = min(minimum_distance, float(info["distance"]))
            if terminated or truncated:
                break
        successes += int(bool(info["is_success"]))
        minimum_distances.append(minimum_distance)
    return successes / episodes, float(np.mean(minimum_distances))


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.steps_per_update <= 0:
        raise ValueError("updates and steps-per-update must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = PPOConfig(
        update_epochs=2 if args.smoke_test else 8,
        minibatch_size=32 if args.smoke_test else 128,
    )
    env = InspireReachEnv(maximum_episode_steps=80 if args.smoke_test else 120)
    evaluation_env = InspireReachEnv(
        maximum_episode_steps=80 if args.smoke_test else 120
    )
    agent = PPOAgent(19, 4, config, args.device)
    observation, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_minimum_distance = float("inf")
    episode_number = 1
    episode_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    best_score = (-1.0, float("-inf"))
    best_update = 0

    try:
        for update in range(1, args.updates + 1):
            rollout: dict[str, list[object]] = {
                key: [] for key in (
                    "observations", "actions", "log_probs", "values",
                    "rewards", "next_values", "terminated", "episode_ends",
                )
            }
            completed_successes: list[int] = []
            for _ in range(args.steps_per_update):
                action, log_prob, value = agent.select_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                episode_end = terminated or truncated
                next_value = 0.0 if terminated else agent.value(next_observation)

                rollout["observations"].append(observation)
                rollout["actions"].append(action)
                rollout["log_probs"].append(log_prob)
                rollout["values"].append(value)
                rollout["rewards"].append(reward)
                rollout["next_values"].append(next_value)
                rollout["terminated"].append(terminated)
                rollout["episode_ends"].append(episode_end)

                episode_reward += reward
                episode_minimum_distance = min(
                    episode_minimum_distance, float(info["distance"])
                )
                observation = next_observation
                if episode_end:
                    success = int(bool(info["is_success"]))
                    completed_successes.append(success)
                    episode_rows.append(
                        {
                            "episode": episode_number,
                            "update": update,
                            "reward": episode_reward,
                            "success": success,
                            "steps": int(info["episode_step"]),
                            "minimum_distance": episode_minimum_distance,
                            "target_x": float(info["target_position"][0]),
                            "target_y": float(info["target_position"][1]),
                            "target_z": float(info["target_position"][2]),
                        }
                    )
                    episode_number += 1
                    episode_reward = 0.0
                    episode_minimum_distance = float("inf")
                    observation, _ = env.reset()

            arrays = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in rollout.items()
            }
            advantages, returns = generalized_advantage_estimate(
                arrays["rewards"], arrays["values"], arrays["next_values"],
                arrays["terminated"], arrays["episode_ends"],
                config.gamma, config.gae_lambda,
            )
            metrics = agent.update(
                {
                    "observations": arrays["observations"],
                    "actions": arrays["actions"],
                    "log_probs": arrays["log_probs"],
                    "advantages": advantages,
                    "returns": returns,
                }
            )
            success_rate = (
                float(np.mean(completed_successes)) if completed_successes else float("nan")
            )
            evaluation_success_rate = float("nan")
            evaluation_mean_minimum_distance = float("nan")
            evaluation_interval = 1 if args.smoke_test else 5
            if update % evaluation_interval == 0:
                evaluation_success_rate, evaluation_mean_minimum_distance = evaluate_policy(
                    agent,
                    evaluation_env,
                    2 if args.smoke_test else 20,
                    args.seed + 10_000,
                )
                score = (
                    evaluation_success_rate,
                    -evaluation_mean_minimum_distance,
                )
                if score > best_score:
                    best_score = score
                    best_update = update
                    agent.save(
                        args.model,
                        {
                            "algorithm": "PPO Actor-Critic",
                            "seed": args.seed,
                            "selected_update": update,
                            "validation_success_rate": evaluation_success_rate,
                            "validation_mean_minimum_distance": evaluation_mean_minimum_distance,
                        },
                    )
            update_rows.append(
                {
                    "update": update,
                    "environment_steps": update * args.steps_per_update,
                    "episodes_completed": len(completed_successes),
                    "success_rate": success_rate,
                    "evaluation_success_rate": evaluation_success_rate,
                    "evaluation_mean_minimum_distance": evaluation_mean_minimum_distance,
                    **metrics,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if update == 1 or update % 5 == 0:
                print(
                    f"update={update:4d} steps={update * args.steps_per_update:7d} "
                    f"success={success_rate:.1%} actor_loss={metrics['actor_loss']:+.4f} "
                    f"critic_loss={metrics['critic_loss']:.4f} "
                    f"validation={evaluation_success_rate:.1%}"
                )
    finally:
        env.close()
        evaluation_env.close()

    metadata = {
        "algorithm": "PPO Actor-Critic",
        "seed": args.seed,
        "updates": args.updates,
        "steps_per_update": args.steps_per_update,
        "environment_steps": args.updates * args.steps_per_update,
        "observation_definition": "q[4], qdot[4], fingertip[3], target[3], delta[3], distance, hold_progress",
        "action_definition": "bounded joint-position increments for shoulder pitch/roll/yaw and elbow",
        "training_seconds": time.perf_counter() - started,
        "selected_update": best_update,
        "validation_success_rate": best_score[0],
        "validation_mean_minimum_distance": -best_score[1],
    }
    write_csv(args.results / "training_updates.csv", update_rows)
    write_csv(args.results / "training_episodes.csv", episode_rows)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "training_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if best_update == 0:
        agent.save(args.model, metadata)
    print(f"Saved checkpoint to {args.model}")
    print(f"Saved metrics to {args.results}")


if __name__ == "__main__":
    main()
