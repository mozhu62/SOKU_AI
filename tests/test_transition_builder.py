from __future__ import annotations

import json

import numpy as np

from soku_ai.data.normalization import NormalizationStats
from soku_ai.data.schemas import OBSERVATION_SHAPE, SCHEMA_VERSION
from soku_ai.data.transition_builder import ProcessedReplayDataset


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


def _write_shard(path, action_shift: int = 1) -> None:
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "replay_id": "r",
                    "schema_version": SCHEMA_VERSION,
                    "action_shift": action_shift,
                    "max_objects_per_side": 4,
                }
            )
        ),
        episode_start=np.asarray([0, 0, 2, 2], np.int32),
        episode_end=np.asarray([2, 2, 4, 4], np.int32),
        state_continuous=np.zeros((4, OBSERVATION_SHAPE.state_continuous), np.float32),
        state_categorical=np.zeros((4, OBSERVATION_SHAPE.state_categorical), np.int64),
        history_numerical=np.ones((4, OBSERVATION_SHAPE.history_numerical), np.float32),
        history_categorical=np.zeros((4, OBSERVATION_SHAPE.history_categorical), np.int64),
        actions=np.asarray([1, 0, 2, 0], np.int64),
        rewards=np.asarray([1, 0, -1, 0], np.float32),
        dones=np.asarray([True, False, True, False]),
        transition_valid=np.asarray([True, False, True, False]),
        self_object_numerical=np.ones((1, OBSERVATION_SHAPE.object_numerical), np.float32),
        self_object_categorical=np.zeros((1, OBSERVATION_SHAPE.object_categorical), np.int64),
        self_object_offsets=np.asarray([0, 1, 1, 1, 1], np.int64),
        opponent_object_numerical=np.empty((0, OBSERVATION_SHAPE.object_numerical), np.float32),
        opponent_object_categorical=np.empty((0, OBSERVATION_SHAPE.object_categorical), np.int64),
        opponent_object_offsets=np.zeros(5, np.int64),
    )


def test_history_does_not_cross_round_and_object_padding(tmp_path) -> None:
    _write_shard(tmp_path / "r.npz")
    dataset = ProcessedReplayDataset(
        tmp_path,
        ["r"],
        _normalization(),
        history_len=4,
        max_objects_per_side=4,
        action_shift=1,
        gamma=0.99,
        n_step=3,
    )
    first = dataset[0]
    second_round = dataset[1]
    assert first["observation"]["history_mask"].sum().item() == 1
    assert second_round["observation"]["history_mask"].sum().item() == 1
    assert first["observation"]["self_object_mask"].tolist() == [True, False, False, False]
    assert not first["observation"]["opponent_object_mask"].any()


def test_action_shift_zero_excludes_current_input(tmp_path) -> None:
    _write_shard(tmp_path / "r.npz", action_shift=0)
    dataset = ProcessedReplayDataset(
        tmp_path,
        ["r"],
        _normalization(),
        history_len=4,
        max_objects_per_side=4,
        action_shift=0,
        gamma=0.99,
        n_step=3,
    )
    assert not dataset[0]["observation"]["history_mask"].any()
