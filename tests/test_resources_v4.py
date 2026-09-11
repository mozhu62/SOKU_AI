"""资源版数据适配的回归用例。本次交付仅添加测试源码，不自动执行。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from soku_ai.config import load_config
from soku_ai.data.action_space import decode_action, encode_action
from soku_ai.data.resources_dataset import ResourceReplayDataset
from soku_ai.data.resources_reader import read_resource_shard
from soku_ai.data.resources_schema import (
    DATASET_SCHEMA, STATE_FIELDS, CATEGORY_FIELDS, TACTICAL_FIELDS, OBJECT_FIELDS, BUTTON_FIELDS,
)
from soku_ai.data.resources_setup import ResourceNormalization, prepare_resources
from soku_ai.inference.export import example_inputs
from soku_ai.models.factory import build_model
from soku_ai.training.checkpoint import load_checkpoint


def _config(root):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/dqfd_suika_resources_v4.yaml")
    config["data"].update(source_dir=str(root / "source"), source_split_file=str(root / "split.json"),
                          split_manifest=str(root / "derived/split.json"), normalization_file=str(root / "derived/norm.json"))
    return config


def _raw(count=6):
    state = np.zeros((count, 18), np.float32)
    state[:, [4, 11]] = 1
    state[:, 16:] = 10000
    result = {
        "metadata_json": np.asarray(json.dumps({
            "dataset_schema": DATASET_SCHEMA, "state_continuous": list(STATE_FIELDS),
            "state_categorical": list(CATEGORY_FIELDS), "tactical_state": list(TACTICAL_FIELDS),
            "object_numerical": list(OBJECT_FIELDS), "object_categorical": ["action", "action_block_id"],
            "button_features": list(BUTTON_FIELDS), "continuous_normalized": False, "action_shift": 1,
        })),
        "state_continuous": state, "state_categorical": np.zeros((count, 5), np.int32),
        "tactical_state": np.zeros((count, 9), np.int8), "episode_id": np.zeros(count, np.int32),
        "transition_valid": np.r_[np.ones(count - 1, bool), False], "terminated": np.zeros(count, bool),
        "action_horizontal": np.zeros(count, np.int8), "action_vertical": np.zeros(count, np.int8),
        "action_buttons": np.zeros((count, 6), np.int8), "action_duration": np.ones(count, np.int16),
    }
    for side in ("self", "opponent"):
        result.update({f"{side}_object_numerical": np.empty((0, 8), np.float32),
                       f"{side}_object_categorical": np.empty((0, 2), np.int32),
                       f"{side}_object_offsets": np.zeros(count + 1, np.int64)})
    return result


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **raw)


def _identity_normalization():
    payload = {}
    for group, size in (("state", 27), ("history", 11), ("object", 8)):
        payload.update({f"{group}_mean": [0.0] * size, f"{group}_std": [1.0] * size})
    return ResourceNormalization(payload)


def _dataset(config, path, shard):
    prepared = {"manifest": {"splits": {"train": [path.name]}}, "normalization": _identity_normalization(),
                "indices": {path.name: np.flatnonzero(shard["transition_valid"]).astype(np.int32)}}
    return ResourceReplayDataset(config, prepared, "train")


@pytest.mark.parametrize("facing", [1, -1])
def test_all_144_source_actions_follow_existing_dqfd_mapping(tmp_path, facing):
    config, raw = _config(tmp_path), _raw(145)
    raw["state_continuous"][:, 4] = facing
    for index in range(144):
        action = decode_action(index)
        h = {0: 0, 1: facing, 2: -facing}[int(action.horizontal)]
        v = {0: 0, 1: -1, 2: 1}[int(action.vertical)]
        raw["action_horizontal"][index], raw["action_vertical"][index] = h, v
        raw["action_buttons"][index] = [action.a, action.d, action.b, action.c, 1, 1]
        assert encode_action(action.horizontal, action.vertical, action.a, action.b, action.c, action.d) == index
    path = tmp_path / "source/actions.npz"
    _write(path, raw)
    shard = read_resource_shard(path, config)
    np.testing.assert_array_equal(shard["actions"][:144], np.arange(144))
    assert int(shard["ignored_card_frames"]) == 144


@pytest.mark.parametrize("shift", [0, 1])
def test_history_uses_previous_aligned_label_and_stops_at_gaps(tmp_path, shift):
    config, raw = _config(tmp_path), _raw(7)
    config["data"]["action_shift"] = shift
    metadata = json.loads(str(raw["metadata_json"].item()))
    metadata["action_shift"] = shift
    raw["metadata_json"] = np.asarray(json.dumps(metadata))
    raw["action_horizontal"][:] = [1, -1, 0, 1, -1, 0, 1]
    raw["action_buttons"][:, 0] = [1, 0, 0, 1, 0, 1, 0]
    raw["action_duration"][:] = np.arange(1, 8)
    raw["transition_valid"][2] = False
    raw["episode_id"][5:] = 1
    path = tmp_path / "source/history.npz"
    _write(path, raw)
    shard = read_resource_shard(path, config)
    np.testing.assert_array_equal(shard["history_valid"], [False, True, True, False, True, False, True])
    np.testing.assert_array_equal(shard["history_numerical"][1, :7], [1, 0, 1, 0, 0, 0, 1])
    dataset = _dataset(config, path, shard)
    num, cat, mask = dataset._history(shard, 2)
    assert mask.sum() == 2
    assert num[-1, 0] == -1
    assert num[-1, 6] == 2
    assert (cat[~mask] == -2).all()
    assert dataset._history(shard, 3)[2].sum() == 0
    assert dataset._history(shard, 4)[2].sum() == 1
    assert dataset._history(shard, 5)[2].sum() == 0
    item = dataset[0]
    assert item["observation"]["state_continuous"].shape == (27,)
    assert item["observation"]["self_object_numerical"].shape == (3, 8)
    assert not item["observation"]["self_object_mask"].any()


def test_hp_rewards_n_step_terminal_and_truncation(tmp_path):
    config, raw = _config(tmp_path), _raw(6)
    raw["state_continuous"][:, 16] = [1000, 950, 950, 900, 1000, 1000]
    raw["state_continuous"][:, 17] = [1000, 900, 700, 0, 1000, 1000]
    raw["episode_id"][4:] = 1
    raw["terminated"][2] = True
    path = tmp_path / "source/rewards.npz"
    _write(path, raw)
    shard = read_resource_shard(path, config)
    np.testing.assert_allclose(shard["rewards"][:3], [.05, .2, 10.65], rtol=1e-6)
    dataset = _dataset(config, path, shard)
    reward, discount, terminal, state = dataset._n_step(shard, 0)
    assert reward == pytest.approx(.05 + .99 * .2 + .99**2 * 10.65)
    assert discount == pytest.approx(.99**3)
    assert terminal and state == 3
    assert not shard["history_valid"][3]
    assert not shard["history_valid"][4]
    shard["transition_valid"][1] = False
    reward, discount, terminal, state = dataset._n_step(shard, 0)
    assert reward == pytest.approx(.05) and discount == pytest.approx(.99)
    assert not terminal and state == 1


def test_existing_split_reused_and_only_train_fits_normalization(tmp_path):
    config = _config(tmp_path)
    root = Path(config["data"]["source_dir"])
    files = {}
    for name, x in (("train.npz", 10), ("val.npz", 900)):
        raw = _raw()
        raw["state_continuous"][:, 0] = x
        _write(root / name, raw)
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    split = {"version": 1, "seed": 1, "train_fraction": .8, "files": files,
             "train": ["train.npz"], "validation": ["val.npz"]}
    split["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    source_split = Path(config["data"]["source_split_file"])
    source_split.write_text(json.dumps(split), encoding="utf-8")
    original = source_split.read_bytes()
    prepared = prepare_resources(config)
    assert prepared["manifest"]["splits"]["train"] == ["train.npz"]
    assert prepared["manifest"]["splits"]["validation"] == ["val.npz"]
    assert prepared["normalization"].payload["state_mean"][0] == 10
    assert prepare_resources(config) is prepared
    assert source_split.read_bytes() == original
    for name, digest in files.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_new_network_dimensions_and_explicit_old_checkpoint_rejection(tmp_path):
    config = _config(tmp_path)
    model = build_model(config).eval()
    assert model.state_encoder.network[0].in_features == 115
    assert model.temporal_encoder.input_projection.in_features == 91
    assert model.object_encoder.entity_network[0].in_features == 48
    assert not hasattr(model.embeddings, "character")
    assert not hasattr(model.embeddings, "stage")
    fields = ("state_continuous", "state_categorical", "history_numerical", "history_categorical", "history_mask",
              "self_object_numerical", "self_object_categorical", "self_object_mask",
              "opponent_object_numerical", "opponent_object_categorical", "opponent_object_mask")
    with torch.no_grad():
        q = model(dict(zip(fields, example_inputs(config, torch.device("cpu")))))
    assert q.shape == (1, 144) and torch.isfinite(q).all()
    path = tmp_path / "old.pt"
    # 故意不提供权重，确保版本检查在 load_state_dict 之前拒绝，不是靠尺寸碰巧报错。
    torch.save({"checkpoint_version": 1, "model_architecture_version": "tcn_entity_dueling_dqn_v1",
                "observation_schema_version": "suika_observation_v1"}, path)
    with pytest.raises(ValueError, match="输入/网络版本不兼容"):
        load_checkpoint(path, online_network=model)
