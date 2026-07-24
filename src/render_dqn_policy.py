from __future__ import annotations

import argparse
from pathlib import Path
import time

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a saved greedy DQN policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goals", type=float, nargs="+", default=[-0.8, 0.8])
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause before and after each episode for a recorded demonstration.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Change goals without resetting the physical robot after the "
            "first segment. Intended for the rendered video only."
        ),
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Additional delay in seconds after each environment step.",
    )
    args = parser.parse_args()
    if args.step_delay < 0:
        raise ValueError("--step-delay must be zero or greater")
    agent, metadata = DQNAgent.load(args.checkpoint, args.device)
    agent.online_network.eval()
    print(f"Loaded {args.checkpoint} ({metadata.get('config_name', 'unknown')})")

    # Reuse one environment and one macOS Viewer for every target. Creating a
    # second passive Viewer immediately after closing the first can race with
    # the macOS UI thread and raise "another MuJoCo viewer is already open".
    env = G1ElbowTargetEnv(render_mode="human")
    try:
        for index, goal in enumerate(args.goals):
            if index == 0 or not args.continuous:
                observation, info = env.reset(
                    seed=args.seed + index,
                    options={"goal_angle": goal},
                )
            else:
                # Preserve qpos, qvel, and controller_target so movement begins
                # from the previous goal. Only the goal-specific episode
                # counters are restarted for this visual demonstration segment.
                env.goal_angle = float(goal)
                env.episode_step = 0
                env.success_streak = 0
                observation = env._get_observation()
                info = env._get_info()
                env.render()
            if args.interactive:
                input(
                    f"Viewer ready for goal {goal:+.2f} rad. "
                    "Position the camera, then press Enter to start..."
                )
            total_reward = 0.0
            while True:
                action = agent.select_action(observation, epsilon=0.0)
                observation, reward, terminated, truncated, info = env.step(action)
                if args.step_delay > 0:
                    time.sleep(args.step_delay)
                total_reward += reward
                print(
                    f"goal={goal:+.2f} step={info['episode_step']:3d} "
                    f"action={action} angle={info['elbow_angle']:+.4f} "
                    f"error={info['angle_error']:+.4f} reward={reward:+.3f}"
                )
                if terminated or truncated:
                    break
            result_label = "SEGMENT" if args.continuous else "RESULT"
            print(
                f"{result_label} goal={goal:+.2f} success={info['is_success']} "
                f"steps={info['episode_step']} total_reward={total_reward:+.3f}"
            )
            if args.interactive:
                next_action = (
                    "continue to the next target"
                    if index + 1 < len(args.goals)
                    else "close the viewer"
                )
                input(
                    f"{result_label.title()} complete. Review the "
                    f"{result_label} line, then press "
                    f"Enter to {next_action}..."
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
