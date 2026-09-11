from __future__ import annotations

import torch
from torch import nn


def combine_dueling_streams(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DuelingQHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return combine_dueling_streams(self.value_stream(values), self.advantage_stream(values))

