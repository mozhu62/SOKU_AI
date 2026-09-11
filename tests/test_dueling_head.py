from __future__ import annotations

import torch

from soku_ai.models.dueling_head import combine_dueling_streams


def test_dueling_formula() -> None:
    value = torch.tensor([[2.0]])
    advantage = torch.tensor([[1.0, 2.0, 3.0]])
    expected = value + advantage - advantage.mean(dim=1, keepdim=True)
    assert torch.allclose(combine_dueling_streams(value, advantage), expected)

