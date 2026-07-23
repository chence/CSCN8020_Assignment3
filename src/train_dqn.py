from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from dqn import DQNAgent, DQNConfig
from experiment_utils import set_global_seeds, write_csv
from g1_rl import G1ElbowTargetEnv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the G1 elbow DQN.")
    parser.add_argument("--config-name", choices=("config_a", "config_b"), required=True)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-hours", type=float, default=2.4)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.episodes <= 0 or args.max_hours <= 0:
        raise ValueError("episodes and max-hours must be positive")
    expected_decay = 0.995 if args.config_name == "config_a" else 0.985
    epsilon_decay = expected_decay if args.epsilon_decay is None else args.epsilon_decay
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon-decay must be in (0, 1]")

    set_global_seeds(args.seed)
    dqn_config = DQNConfig(
        warmup_transitions=64 if args.smoke_test else 500,
        target_update_steps=10 if args.smoke_test else 250,
    )
    env = G1ElbowTargetEnv(goal_range=(-0.8, 0.8))
    agent = DQNAgent(4, 3, dqn_config, args.seed, args.device)
    output_dir = args.output_root / args.config_name
    checkpoint_path = args.model_root / f"{args.config_name}.pt"
    rows: list[dict[str, float | int]] = []
    recent_successes: deque[int] = deque(maxlen=50)
    epsilon = args.epsilon_start
    started = time.perf_counter()

    try:
        for episode in range(1, args.episodes + 1):
            if (time.perf_counter() - started) / 3600.0 >= args.max_hours:
                print("Stopping at the configured wall-clock limit.")
                break
            observation, info = env.reset(seed=args.seed + episode - 1)
            total_reward = 0.0
            losses: list[float] = []
            while True:
                action = agent.select_action(observation, epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                # Only true success termination masks the Bellman bootstrap.
                agent.remember(
                    observation, action, reward, next_observation, terminated
                )
                loss = agent.optimize_model()
                if loss is not None:
                    losses.append(loss)
                observation = next_observation
                total_reward += reward
                if terminated or truncated:
                    break

            success = int(bool(info["is_success"]))
            recent_successes.append(success)
            rows.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode - 1,
                    "goal": float(info["goal_angle"]),
                    "cumulative_reward": total_reward,
                    "success": success,
                    "episode_length": int(info["episode_step"]),
                    "final_absolute_error": float(info["absolute_error"]),
                    "epsilon": epsilon,
                    "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                    "optimization_steps": agent.optimization_steps,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            epsilon = max(args.epsilon_min, epsilon * epsilon_decay)
            if episode == 1 or episode % 10 == 0:
                print(
                    f"episode={episode:4d} reward={total_reward:+8.3f} "
                    f"success={success} rolling_success={np.mean(recent_successes):.2%} "
                    f"epsilon={epsilon:.4f}"
                )
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    metadata = {
        "config_name": args.config_name,
        "seed": args.seed,
        "epsilon_decay": epsilon_decay,
        "epsilon_final": epsilon,
        "episodes_completed": len(rows),
        "training_seconds": elapsed,
        "goal_range": [-0.8, 0.8],
        "truncation_treatment": "bootstrap across time-limit truncation",
    }
    write_csv(output_dir / "training_metrics.csv", rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    agent.save(checkpoint_path, metadata)
    print(f"Saved metrics to {output_dir}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
