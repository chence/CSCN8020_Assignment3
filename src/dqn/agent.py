from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from dqn.q_network import QNetwork
from dqn.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_transitions: int = 500
    target_update_steps: int = 250
    gradient_clip_norm: float = 10.0
    hidden_size: int = 64


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DQNConfig,
        seed: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.action_size = action_size
        self.device = torch.device(device)
        self._random = random.Random(seed)

        self.online_network = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(self.device)
        self.target_network = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=config.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(config.replay_capacity, seed)
        self.optimization_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if self._random.random() < epsilon:
            return self._random.randrange(self.action_size)
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_network(observation_tensor)
            return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.replay_buffer.add(
            observation, action, reward, next_observation, terminated
        )

    def optimize_model(self) -> float | None:
        required = max(
            self.config.batch_size, self.config.warmup_transitions
        )
        if len(self.replay_buffer) < required:
            return None

        batch = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )
        selected_q_values = self.online_network(batch.observations).gather(
            1, batch.actions
        )

        with torch.no_grad():
            next_q_values = self.target_network(
                batch.next_observations
            ).max(dim=1, keepdim=True).values
            # True terminal states do not bootstrap. Time-limit truncations
            # are intentionally stored as non-terminal transitions.
            targets = batch.rewards + (
                self.config.gamma * (~batch.terminated).float() * next_q_values
            )

        loss = self.loss_function(selected_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.optimization_steps += 1

        if self.optimization_steps % self.config.target_update_steps == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "observation_size": self.online_network.network[0].in_features,
                "action_size": self.action_size,
                "optimization_steps": self.optimization_steps,
                "metadata": metadata,
            },
            output_path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["DQNAgent", dict[str, Any]]:
        checkpoint = torch.load(
            Path(path), map_location=torch.device(device), weights_only=False
        )
        agent = cls(
            observation_size=int(checkpoint["observation_size"]),
            action_size=int(checkpoint["action_size"]),
            config=DQNConfig(**checkpoint["config"]),
            seed=int(checkpoint.get("metadata", {}).get("seed", 0)),
            device=device,
        )
        agent.online_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.optimization_steps = int(checkpoint["optimization_steps"])
        return agent, dict(checkpoint.get("metadata", {}))
