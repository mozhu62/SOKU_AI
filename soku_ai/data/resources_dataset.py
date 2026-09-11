from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .resources_reader import read_resource_shard
from .transition_builder import ProcessedReplayDataset, CATEGORICAL_PADDING_VALUE


class ResourceReplayDataset(ProcessedReplayDataset):
    """复用原 DQfD Transition 契约；直接读取 v4，按字节预算缓存归一化后的分片。"""

    def __init__(self, config, prepared, split):
        self.config = config
        self.replay_ids = tuple(prepared["manifest"]["splits"][split])
        self.shard_paths = tuple(Path(config["data"]["source_dir"]) / name for name in self.replay_ids)
        self.normalization = prepared["normalization"]
        self.history_len = int(config["data"]["history_len"])
        self.max_objects = 3
        self.gamma, self.n_step = float(config["rl"]["gamma"]), int(config["rl"]["n_step"])
        self.index_shards = np.concatenate([np.full(len(prepared["indices"][name]), i, np.int32)
                                            for i, name in enumerate(self.replay_ids)]) if self.replay_ids else np.empty(0, np.int32)
        self.index_frames = np.concatenate([prepared["indices"][name] for name in self.replay_ids]) if self.replay_ids else np.empty(0, np.int32)
        self._cache = OrderedDict()
        self.cache_bytes = 0
        self.budget = int(float(config["data"].get("cache_gb", 4)) * 1024 ** 3)
        self.include_evaluation_metadata = False
        self.evaluation_metadata = split != "train"

    def _load_shard(self, shard_index):
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]
        shard = read_resource_shard(self.shard_paths[shard_index], self.config)
        for key, group in (("state_continuous", "state"), ("history_numerical", "history"),
                           ("self_object_numerical", "object"), ("opponent_object_numerical", "object")):
            shard[key] = self.normalization.normalize(shard[key], group)
        size = sum(v.nbytes for v in shard.values())
        while self._cache and self.cache_bytes + size > self.budget:
            _, old = self._cache.popitem(last=False)
            self.cache_bytes -= sum(v.nbytes for v in old.values())
        if size <= self.budget:
            self._cache[shard_index] = shard
            self.cache_bytes += size
        return shard

    def preload_shards(self):
        for index in range(len(self.shard_paths)):
            self._load_shard(index)
            if len(self._cache) != index + 1:
                raise ValueError("cache_gb 不足以容纳全部分片；关闭 preload_shards 使用有界缓存，或提高预算")

    def _history(self, shard, frame_index):
        end = frame_index + 1
        start = max(int(shard["episode_start"][frame_index]), end - self.history_len)
        rows = np.arange(start, end)
        mask_source = shard["history_valid"][rows]
        count = len(rows)
        numerical = np.zeros((self.history_len, 11), np.float32)
        categorical = np.full((self.history_len, 4), CATEGORICAL_PADDING_VALUE, np.int64)
        mask = np.zeros(self.history_len, bool)
        positions = np.arange(self.history_len - count, self.history_len)[mask_source]
        numerical[positions] = shard["history_numerical"][rows[mask_source]]
        categorical[positions] = shard["history_categorical"][rows[mask_source]]
        mask[positions] = True
        return numerical, categorical, mask

    def _objects(self, shard, frame_index, side):
        offsets = shard[f"{side}_object_offsets"]
        start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
        count = end - start
        numerical = np.zeros((3, 8), np.float32)
        categorical = np.full((3, 2), CATEGORICAL_PADDING_VALUE, np.int64)
        numerical[:count] = shard[f"{side}_object_numerical"][start:end]
        categorical[:count] = shard[f"{side}_object_categorical"][start:end]
        return numerical, categorical, np.arange(3) < count

    def _n_step(self, shard, start_index):
        reward, discount, terminal, final_state = 0.0, 1.0, False, start_index
        for index in range(start_index, min(start_index + self.n_step, len(shard["actions"]) - 1)):
            if not shard["transition_valid"][index]:
                break
            reward += discount * float(shard["rewards"][index])
            discount *= self.gamma
            final_state, terminal = index + 1, bool(shard["dones"][index])
            if terminal:
                break
        # 真实终局禁止 bootstrap；断帧/截断不伪造成输局，仍从最后真实可用状态 bootstrap。
        return reward, discount, terminal, final_state

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.evaluation_metadata:
            shard = self._load_shard(int(self.index_shards[index]))
            frame = int(self.index_frames[index])
            result.update(active_weather=torch.tensor(int(shard["state_categorical"][frame, 4]), dtype=torch.long),
                          object_count=torch.tensor(sum(int(shard[f"{side}_object_offsets"][frame+1] -
                                                            shard[f"{side}_object_offsets"][frame])
                                                        for side in ("self", "opponent")), dtype=torch.long))
        return result
