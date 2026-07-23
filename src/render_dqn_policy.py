from __future__ import annotations

import argparse
from pathlib import Path

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a saved greedy DQN policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--goals", type=float, nargs="+", default=[-0.8, 0.8])
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    agent, metadata = DQNAgent.load(args.checkpoint, args.device)
    agent.online_network.eval()
    print(f"Loaded {args.checkpoint} ({metadata.get('config_name', 'unknown')})")

    for index, goal in enumerate(args.goals):
        env = G1ElbowTargetEnv(render_mode="human", goal_angle=goal)
        try:
            observation, info = env.reset(seed=args.seed + index)
            total_reward = 0.0
            while True:
                action = agent.select_action(observation, epsilon=0.0)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(
                    f"goal={goal:+.2f} step={info['episode_step']:3d} "
                    f"action={action} angle={info['elbow_angle']:+.4f} "
                    f"error={info['angle_error']:+.4f} reward={reward:+.3f}"
                )
                if terminated or truncated:
                    break
            print(
                f"RESULT goal={goal:+.2f} success={info['is_success']} "
                f"steps={info['episode_step']} total_reward={total_reward:+.3f}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
