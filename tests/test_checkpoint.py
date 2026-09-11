from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from soku_ai.data.normalization import NormalizationStats
from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.data.split import manifest_hash, save_split_manifest
from soku_ai.env.mock_env import empty_observation
from soku_ai.models.factory import build_model
from soku_ai.training.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        np.zeros(OBSERVATION_SHAPE.state_continuous, np.float32),
        np.ones(OBSERVATION_SHAPE.state_continuous, np.float32),
        np.zeros(OBSERVATION_SHAPE.history_numerical, np.float32),
        np.ones(OBSERVATION_SHAPE.history_numerical, np.float32),
        np.zeros(OBSERVATION_SHAPE.object_numerical, np.float32),
        np.ones(OBSERVATION_SHAPE.object_numerical, np.float32),
        ("r",),
    )


def test_checkpoint_round_trip_preserves_q(tmp_path, model_config) -> None:
    config = deepcopy(model_config)
    manifest = {
        "version": 1,
        "seed": 1,
        "ratios": {"train": 1.0, "validation": 0.0, "test": 0.0},
        "splits": {"train": ["r"], "validation": [], "test": []},
    }
    manifest["sha256"] = manifest_hash(manifest)
    manifest_path = tmp_path / "split.json"
    save_split_manifest(manifest, manifest_path)
    config["data"]["split_manifest"] = str(manifest_path)
    online = build_model(config).eval()
    target = build_model(config).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    observation = empty_observation(32, 32)
    with torch.no_grad():
        expected = online(observation).clone()
    checkpoint = build_checkpoint(
        online_network=online,
        target_network=target,
        optimizer=optimizer,
        config=config,
        normalization=_normalization(),
        step=12,
        epoch=1,
        seed=1,
        best_metric=0.5,
    )
    path = tmp_path / "model.pt"
    save_checkpoint(checkpoint, path)
    restored = build_model(config).eval()
    load_checkpoint(path, online_network=restored)
    with torch.no_grad():
        actual = restored(observation)
    assert torch.equal(expected, actual)

