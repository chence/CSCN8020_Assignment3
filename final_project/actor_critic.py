from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    update_epochs: int = 8
    minibatch_size: int = 128
    gradient_clip_norm: float = 0.5
    hidden_size: int = 128


class ActorCritic(nn.Module):
    """Gaussian Actor and state-value Critic used by PPO."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def _distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observations)
        std = self.log_std.clamp(-4.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.square() + 1e-6)
        return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

    def act(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions)
        return actions, log_prob, self.critic(observations).squeeze(-1)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_actions = actions.clamp(-0.999999, 0.999999)
        raw_actions = torch.atanh(safe_actions)
        distribution = self._distribution(observations)
        log_prob = self._squashed_log_prob(distribution, raw_actions, safe_actions)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(observations).squeeze(-1)
        return log_prob, entropy, values


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: PPOConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.network = ActorCritic(
            observation_size, action_size, config.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.observation_size = observation_size
        self.action_size = action_size
        self.updates = 0

    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.network.act(
                observation_tensor, deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, observation: np.ndarray) -> float:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return float(self.network.critic(tensor).item())

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            batch["log_probs"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            batch["returns"], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            batch["advantages"], dtype=torch.float32, device=self.device
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics: list[tuple[float, float, float]] = []
        sample_count = len(observations)
        for _ in range(self.config.update_epochs):
            for indices in torch.randperm(sample_count, device=self.device).split(
                self.config.minibatch_size
            ):
                log_probs, entropy, values = self.network.evaluate_actions(
                    observations[indices], actions[indices]
                )
                ratio = (log_probs - old_log_probs[indices]).exp()
                unclipped = ratio * advantages[indices]
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = 0.5 * (returns[indices] - values).square().mean()
                entropy_mean = entropy.mean()
                loss = (
                    actor_loss
                    + self.config.value_coefficient * critic_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip_norm
                )
                self.optimizer.step()
                metrics.append(
                    (float(actor_loss.item()), float(critic_loss.item()), float(entropy_mean.item()))
                )
        self.updates += 1
        means = np.mean(metrics, axis=0)
        return {
            "actor_loss": float(means[0]),
            "critic_loss": float(means[1]),
            "entropy": float(means[2]),
        }

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "updates": self.updates,
                "metadata": metadata,
            },
            output,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["PPOAgent", dict[str, Any]]:
        checkpoint = torch.load(
            Path(path), map_location=torch.device(device), weights_only=False
        )
        agent = cls(
            int(checkpoint["observation_size"]),
            int(checkpoint["action_size"]),
            PPOConfig(**checkpoint["config"]),
            device,
        )
        agent.network.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.updates = int(checkpoint.get("updates", 0))
        return agent, dict(checkpoint.get("metadata", {}))


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_ends: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = 1.0 - float(terminated[index])
        delta = rewards[index] + gamma * next_values[index] * bootstrap - values[index]
        continuation = 1.0 - float(episode_ends[index])
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[index] = gae
    return advantages, advantages + values
