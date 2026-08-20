from __future__ import annotations

import numpy as np

from actor_critic import PPOAgent, PPOConfig, generalized_advantage_estimate
from inspire_reach_env import InspireReachEnv


def test_actor_critic_environment_and_update() -> None:
    env = InspireReachEnv(maximum_episode_steps=3)
    try:
        observation, info = env.reset(seed=7)
        assert observation.shape == (19,)
        assert env.observation_space.contains(observation)
        assert info["target_position"].shape == (3,)
        next_observation, reward, terminated, truncated, next_info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        assert next_observation.shape == (19,)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert np.isfinite(next_info["distance"])

        agent = PPOAgent(19, 4, PPOConfig(update_epochs=1, minibatch_size=4))
        action, log_prob, value = agent.select_action(observation)
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1.0)
        assert np.isfinite([log_prob, value]).all()

        rewards = np.ones(4, dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        advantages, returns = generalized_advantage_estimate(
            rewards, values, values, np.zeros(4), np.array([0, 1, 0, 1]), 0.99, 0.95
        )
        metrics = agent.update(
            {
                "observations": np.repeat(observation[None, :], 4, axis=0),
                "actions": np.repeat(action[None, :], 4, axis=0),
                "log_probs": np.full(4, log_prob, dtype=np.float32),
                "advantages": advantages,
                "returns": returns,
            }
        )
        assert all(np.isfinite(list(metrics.values())))
    finally:
        env.close()


def main() -> None:
    test_actor_critic_environment_and_update()
    print("Actor-Critic environment and PPO update tests passed.")


if __name__ == "__main__":
    main()

