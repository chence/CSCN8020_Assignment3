from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from actor_critic import PPOAgent
from inspire_reach_env import InspireReachEnv


def format_vector(values: np.ndarray, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Inspire PPO Actor-Critic policy."
    )
    parser.add_argument("--model", type=Path, default=Path("models/inspire_actor_critic.pt"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--keep-viewer-open",
        action="store_true",
        help="Keep the MuJoCo viewer open after evaluation finishes.",
    )
    parser.add_argument("--csv", type=Path, default=Path("results/inspire_actor_critic/evaluation.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.model}")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    agent, metadata = PPOAgent.load(args.model, args.device)
    env = InspireReachEnv(render_mode="human" if args.viewer else None)
    rows: list[dict[str, object]] = []
    print("=" * 68)
    print("INSPIRE REACH-AND-TOUCH: SAVED ACTOR-CRITIC EVALUATION")
    print("=" * 68)
    print(f"1. CHECKPOINT LOADED: {args.model}")
    print(f"   algorithm={metadata.get('algorithm', 'PPO Actor-Critic')}")
    print(f"   selected_update={metadata.get('selected_update', 'unknown')}")
    print("2. TARGETS: a new reachable blue target is generated per episode")
    print("3. POLICY: deterministic Actor controls 4 left-arm joints")
    print("   joints=shoulder_pitch, shoulder_roll, shoulder_yaw, elbow")
    print("4. METRICS: success, steps, reward, and fingertip distance")
    print(f"   episodes={args.episodes} viewer={args.viewer}\n")
    try:
        for episode in range(1, args.episodes + 1):
            observation, info = env.reset(seed=args.seed + episode - 1)
            total_reward = 0.0
            minimum_distance = float("inf")
            initial_distance = float(info["distance"])
            print("-" * 68)
            print(f"EPISODE {episode}/{args.episodes}")
            print(
                "NEW BLUE TARGET: "
                f"xyz={format_vector(info['target_position'], 4)}"
            )
            print(
                f"initial_fingertip_distance={initial_distance:.4f}m "
                "actor_mode=deterministic"
            )
            while True:
                action, _, value = agent.select_action(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                minimum_distance = min(minimum_distance, float(info["distance"]))
                step = int(info["episode_step"])
                if step == 1 or step % 10 == 0 or terminated or truncated:
                    print(
                        f"step={step:3d} distance={float(info['distance']):.4f}m "
                        f"critic_V={value:+.3f} "
                        f"action={format_vector(action)}"
                    )
                if terminated or truncated:
                    break
            row = {
                "episode": episode,
                "seed": args.seed + episode - 1,
                "success": int(bool(info["is_success"])),
                "steps": int(info["episode_step"]),
                "reward": total_reward,
                "minimum_distance": minimum_distance,
                "final_distance": float(info["distance"]),
                "target_x": float(info["target_position"][0]),
                "target_y": float(info["target_position"][1]),
                "target_z": float(info["target_position"][2]),
                "final_critic_value": value,
            }
            rows.append(row)
            print(
                f"RESULT: {'TOUCH SUCCESS' if row['success'] else 'TIME LIMIT'} | "
                f"minimum_distance={minimum_distance:.4f}m | "
                f"steps={row['steps']} | reward={total_reward:+.3f}"
            )
        if args.viewer and args.keep_viewer_open and env.viewer is not None:
            print("\nEvaluation complete. Close the MuJoCo viewer to exit.")
            while env.viewer.is_running():
                env.render()
                time.sleep(1.0 / 30.0)
    finally:
        env.close()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(int(row["success"]) for row in rows)
    print("\n" + "=" * 68)
    print("EVALUATION SUMMARY")
    print("=" * 68)
    print(f"algorithm={metadata.get('algorithm', 'PPO Actor-Critic')}")
    print(f"checkpoint={args.model}")
    print(f"success_rate={successes}/{len(rows)} ({successes / len(rows):.1%})")
    print(f"mean_minimum_distance={np.mean([row['minimum_distance'] for row in rows]):.4f}m")
    print(f"mean_episode_reward={np.mean([row['reward'] for row in rows]):+.3f}")
    print(f"csv={args.csv}")



if __name__ == "__main__":
    main()
