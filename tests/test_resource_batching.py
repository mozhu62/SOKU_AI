"""批量路径与原逐条 Dataset 对照；不依赖真实 REP，不启动游戏。"""
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from soku_ai.data.resources_dataset import ResourceReplayDataset
from soku_ai.training.batching import PackedBatchTransfer


def make_dataset(n_step, evaluation):
    rng = np.random.default_rng(20260912)
    shards = []
    for _ in range(2):
        count = 18
        history_valid = np.ones(count, bool)
        history_valid[[0, 5, 10]] = False
        valid = np.ones(count, bool)
        valid[[4, count - 1]] = False
        dones = np.zeros(count, bool)
        dones[9] = True
        shard = {
            "state_continuous": rng.normal(size=(count, 27)).astype(np.float32),
            "state_categorical": rng.integers(0, 100, size=(count, 5), dtype=np.int64),
            "history_numerical": rng.normal(size=(count, 11)).astype(np.float32),
            "history_categorical": rng.integers(0, 100, size=(count, 4), dtype=np.int64),
            "history_valid": history_valid,
            "episode_start": np.maximum.accumulate(np.where(~history_valid, np.arange(count), 0)),
            "actions": rng.integers(0, 144, count, dtype=np.int64),
            "rewards": rng.normal(size=count).astype(np.float32),
            "dones": dones, "transition_valid": valid,
            "training_weights": rng.uniform(.5, 2, count).astype(np.float32),
            "round_outcomes": rng.integers(-1, 2, count, dtype=np.int8),
        }
        for side in ("self", "opponent"):
            offsets = np.r_[0, np.cumsum(np.arange(count) % 4)].astype(np.int64)
            shard.update({f"{side}_object_offsets": offsets,
                          f"{side}_object_numerical": rng.normal(size=(offsets[-1], 8)).astype(np.float32),
                          f"{side}_object_categorical": rng.integers(0, 100, size=(offsets[-1], 2), dtype=np.int64)})
        shards.append(shard)
    # 第二分片还覆盖无可选训练权重/输赢字段的兼容路径。
    del shards[1]["training_weights"], shards[1]["round_outcomes"]
    dataset = ResourceReplayDataset.__new__(ResourceReplayDataset)
    dataset.history_len, dataset.n_step, dataset.gamma = 32, n_step, .99
    dataset.index_shards = np.repeat(np.arange(2), 17)
    dataset.index_frames = np.tile(np.arange(17), 2)
    dataset.include_evaluation_metadata = False
    dataset.evaluation_metadata = evaluation
    dataset._load_shard = shards.__getitem__
    return dataset


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, dict):
            assert_same(value, expected[key])
        else:
            assert value.shape == expected[key].shape
            assert value.dtype == expected[key].dtype
            torch.testing.assert_close(value.cpu(), expected[key].cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("n_step", [1, 3, 40])
@pytest.mark.parametrize("evaluation", [False, True])
def test_batch_equals_scalar_dataset(n_step, evaluation):
    dataset = make_dataset(n_step, evaluation)
    # 非排序、跨分片、重复索引、片段首尾、终局与内部无效转移均须保留旧路径结果。
    indices = np.array([18, 0, 16, 8, 9, 10, 3, 4, 5, 18, 33, 0])
    for selection in (indices, [0], [9], [16], [0, 18]):
        expected = default_collate([dataset[int(index)] for index in selection])
        assert_same(dataset.get_batch(selection), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_transfer_round_trip_and_buffer_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证固定页传输")
    mover = PackedBatchTransfer(torch.device(device))
    dataset = make_dataset(3, True)
    first = dataset.get_batch([0, 1, 18])
    output = mover.transfer(mover.prepare(first))
    second = dataset.get_batch([16, 8, 9, 10])
    next_output = mover.transfer(mover.prepare(second))
    assert_same(output, first)
    assert_same(next_output, second)
