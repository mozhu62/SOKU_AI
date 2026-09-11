from __future__ import annotations

import torch

from soku_ai.env.mock_env import empty_observation
from soku_ai.models.factory import build_model


def _repeat(observation, batch):
    return {name: value.repeat(batch, *([1] * (value.ndim - 1))) for name, value in observation.items()}


def test_q_network_shapes(model_config) -> None:
    model = build_model(model_config)
    base = empty_observation(32, 32)
    for batch in (1, 8, 64):
        output = model(_repeat(base, batch))
        assert output.shape == (batch, 144)
        assert torch.isfinite(output).all()


def test_single_backward_step(model_config) -> None:
    model = build_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model(empty_observation(32, 32))
    loss = output.square().mean()
    loss.backward()
    optimizer.step()

