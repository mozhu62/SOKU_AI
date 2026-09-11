from __future__ import annotations

import pytest
import torch

from soku_ai.env.mock_env import empty_observation
from soku_ai.models.factory import build_model
from soku_ai.training.utils import move_to_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有 CUDA")
def test_cuda_forward_backward(model_config) -> None:
    device = torch.device("cuda")
    model = build_model(model_config).to(device)
    observation = move_to_device(empty_observation(32, 32), device)
    loss = model(observation).square().mean()
    loss.backward()

