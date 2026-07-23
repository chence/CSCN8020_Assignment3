from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True)
class TransitionBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Bounded experience-replay memory with random mini-batches."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            deque(maxlen=capacity)
        )
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self._memory.append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(terminated),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self) < batch_size:
            raise ValueError("not enough transitions for requested batch")

        transitions = self._random.sample(list(self._memory), batch_size)
        observations, actions, rewards, next_observations, terminated = zip(
            *transitions
        )
        return TransitionBatch(
            observations=torch.as_tensor(
                np.stack(observations), dtype=torch.float32, device=device
            ),
            actions=torch.as_tensor(
                actions, dtype=torch.int64, device=device
            ).unsqueeze(1),
            rewards=torch.as_tensor(
                rewards, dtype=torch.float32, device=device
            ).unsqueeze(1),
            next_observations=torch.as_tensor(
                np.stack(next_observations), dtype=torch.float32, device=device
            ),
            terminated=torch.as_tensor(
                terminated, dtype=torch.bool, device=device
            ).unsqueeze(1),
        )
