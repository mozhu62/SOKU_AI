from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from soku_ai.config import load_config


@pytest.fixture()
def model_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "dqfd_suika_v1.yaml"
    config = load_config(path)
    config = deepcopy(config)
    config["model"]["temporal_channels"] = [32, 32, 64, 64]
    config["model"]["state_hidden_dim"] = 64
    config["model"]["fusion_hidden_dim"] = 128
    config["model"]["fused_dim"] = 64
    config["model"]["dueling_hidden_dim"] = 32
    return config

