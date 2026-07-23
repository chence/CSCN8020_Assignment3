from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map the four-value elbow observation to three action values."""

    def __init__(
        self,
        observation_size: int = 4,
        action_size: int = 3,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
