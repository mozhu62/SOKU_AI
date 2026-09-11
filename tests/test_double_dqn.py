from __future__ import annotations

import torch
from torch import nn

from soku_ai.rl.double_dqn import double_dqn_target


class FixedQ(nn.Module):
    def __init__(self, values) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observation):
        return self.values.unsqueeze(0).expand(observation["x"].shape[0], -1)


def test_double_dqn_uses_online_argmax_and_target_value() -> None:
    online = FixedQ([1.0, 5.0, 2.0])
    target = FixedQ([10.0, 20.0, 100.0])
    result = double_dqn_target(
        online,
        target,
        {"x": torch.zeros(2, 1)},
        torch.tensor([1.0, 2.0]),
        torch.tensor([False, True]),
        torch.tensor([0.5, 0.5]),
    )
    assert torch.allclose(result, torch.tensor([11.0, 2.0]))

