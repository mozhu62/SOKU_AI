"""实战/离线观察一致性回归；使用合成帧，不启动游戏或发送系统按键。"""
import json
from pathlib import Path

import numpy as np
import pytest

from soku_ai.config import load_config
from soku_ai.data.resources_schema import (
    DATASET_SCHEMA, OBSERVATION_VERSION, STATE_FIELDS, CATEGORY_FIELDS,
    TACTICAL_FIELDS, HISTORY_FIELDS, OBJECT_FIELDS, BUTTON_FIELDS,
)
from soku_ai.data.resources_setup import ResourceNormalization
from soku_ai.data.resources_reader import read_resource_shard
from soku_ai.data.resources_dataset import ResourceReplayDataset
from soku_ai.live.resources_observation import ResourceLiveObservationBuilder
from soku_ai.live.observation import LiveStateUnavailable
from soku_ai.live.shared_state import StatePayload


def configuration():
    return load_config(Path(__file__).resolve().parents[1] / "configs/dqfd_suika_resources_v4.yaml")


def normalization():
    result = {"version": OBSERVATION_VERSION}
    for group, fields in (("state", STATE_FIELDS + TACTICAL_FIELDS), ("history", HISTORY_FIELDS), ("object", OBJECT_FIELDS)):
        result.update({f"{group}_features": list(fields), f"{group}_mean": [0.] * len(fields),
                       f"{group}_std": [1.] * len(fields)})
    return result


def frames(count):
    result = []
    for i in range(count):
        p = StatePayload()
        p.initialized = p.inBattle = 1
        p.matchState, p.battleMode, p.battleSubMode = 2, 2, 0
        p.gameProcessId, p.currentRound, p.battleFrame, p.sampleSerial = 1, 1, i + 1, i + 1
        for player in (p.left, p.right):
            player.hp, player.currentSpirit, player.direction = 1000, 1000, 1
            player.actionFrameCount, player.frameDataAvailable = i, 1
        p.left.positionX, p.left.speedX, p.right.positionX = i, 2, 100
        p.left.frameFlags, p.right.frameFlags = (1 << 11) | (1 << 2), 1 << 10
        p.left.action, p.right.action = 3, 60
        p.left.inputHorizontal = 1 if (i // 8) % 2 else -1
        p.left.inputA, p.left.inputD = i % 3 == 0, i % 4 == 0
        p.right.objectCount = p.right.totalObjectCount = 4
        for j, dx in enumerate((30, 10, 20, 5)):
            obj = p.right.objects[j]
            obj.positionX, obj.direction, obj.action = i + dx, 1, 400 + j
            obj.frameDataAvailable = 1
            obj.hitBoxes[0].valid = j == 0 or (j == 3 and i % 2 == 1)
        result.append(p)
    return result


@pytest.mark.parametrize("shift", [0, 1])
def test_live_observation_matches_actual_npz_reader_and_dataset(tmp_path, shift):
    config, norm, source_frames = configuration(), normalization(), frames(40)
    config["data"].update(action_shift=shift, source_dir=str(tmp_path))
    count = len(source_frames)
    numeric, cats, tactical, controls = [], [], [], []
    objects, object_cats, duration = [], [], []
    for i, p in enumerate(source_frames):
        numeric.append([i, 0, 2, 0, 1, 1000, i, 100, 0, 0, 0, 1, 1000, i, 100-i, 0, 1000, 1000])
        cats.append([3, 0, 60, 0, 0])
        tactical.append([1, 0, 0, 1, i % 2, 0, 1, 1, 0])
        controls.append([p.left.inputHorizontal, 0, p.left.inputA, p.left.inputD, 0, 0, 0, 0])
        duration.append(i % 8 + 1)
        for j, dx in ((3, 5), (1, 10), (2, 20)):
            objects.append([dx, 0, -2, 0, 1, 0, 0, 0])
            object_cats.append([400+j, 0])
    control = np.asarray(controls, np.int8)
    source = np.minimum(np.arange(count) + shift, count - 1)
    meta = {"dataset_schema": DATASET_SCHEMA, "state_continuous": list(STATE_FIELDS),
            "state_categorical": list(CATEGORY_FIELDS), "tactical_state": list(TACTICAL_FIELDS),
            "object_numerical": list(OBJECT_FIELDS), "object_categorical": ["action", "action_block_id"],
            "button_features": list(BUTTON_FIELDS), "continuous_normalized": False, "action_shift": shift}
    path = tmp_path / "live.npz"
    np.savez(path, metadata_json=np.asarray(json.dumps(meta)), episode_id=np.zeros(count, np.int32),
        state_continuous=np.asarray(numeric, np.float32), state_categorical=np.asarray(cats, np.int64),
        tactical_state=np.asarray(tactical, np.float32), transition_valid=np.r_[np.ones(count-1, bool), False],
        terminated=np.zeros(count, bool), action_horizontal=control[source, 0], action_vertical=control[source, 1],
        action_buttons=control[source, 2:], action_duration=np.asarray(duration, np.int32)[source],
        self_object_numerical=np.empty((0, 8), np.float32), self_object_categorical=np.empty((0, 2), np.int64),
        self_object_offsets=np.zeros(count+1, np.int64), opponent_object_numerical=np.asarray(objects, np.float32),
        opponent_object_categorical=np.asarray(object_cats, np.int64), opponent_object_offsets=np.arange(count+1)*3)
    shard = read_resource_shard(path, config)
    prepared = {"manifest": {"splits": {"train": [path.name]}}, "normalization": ResourceNormalization(norm),
                "indices": {path.name: np.flatnonzero(shard["transition_valid"])}}
    dataset = ResourceReplayDataset(config, prepared, "train")
    builder = ResourceLiveObservationBuilder(config, norm, "left")
    for i, payload in enumerate(source_frames[:-1]):
        live = builder.build(payload)
        expected = dataset[i]["observation"]
        for key, value in expected.items():
            np.testing.assert_allclose(live.observation[key].numpy(), value.numpy(), err_msg=f"frame={i} {key}")
    assert live.history_count == 32


def test_gap_round_terminal_and_duplicate_frames_reset_history():
    builder = ResourceLiveObservationBuilder(configuration(), normalization(), "left")
    payloads = frames(5)
    assert builder.build(payloads[0]).history_count == 0
    assert builder.build(payloads[1]).history_count == 1
    assert builder.build(payloads[1]) is None
    assert builder.build(payloads[3]).history_count == 0
    payloads[4].currentRound = 2
    assert builder.build(payloads[4]).history_count == 0
    payloads[4].left.hp = 0
    with pytest.raises(LiveStateUnavailable, match="本小局已结束"):
        builder.build(payloads[4])
    assert not builder.history_num


def test_normalization_schema_is_checked_before_control():
    norm = normalization()
    norm["history_features"][0] = "current_label_leak"
    with pytest.raises(ValueError, match="字段顺序"):
        ResourceLiveObservationBuilder(configuration(), norm)
