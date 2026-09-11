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

    def _observations_many(self, shard, frames):
        count = len(frames)
        history_rows = frames[:, None] + np.arange(1 - self.history_len, 1)
        inside = history_rows >= shard["episode_start"][frames, None]
        safe_rows = np.maximum(history_rows, 0)
        history_mask = inside & shard["history_valid"][safe_rows]
        observation = {
            "state_continuous": shard["state_continuous"][frames],
            "state_categorical": shard["state_categorical"][frames],
            "history_numerical": np.where(history_mask[..., None], shard["history_numerical"][safe_rows], 0),
            "history_categorical": np.where(history_mask[..., None], shard["history_categorical"][safe_rows], CATEGORICAL_PADDING_VALUE),
            "history_mask": history_mask,
        }
        for side in ("self", "opponent"):
            offsets = shard[f"{side}_object_offsets"]
            starts, ends = offsets[frames], offsets[frames + 1]
            mask = np.arange(3) < (ends - starts)[:, None]
            rows = starts[:, None] + np.arange(3)
            numerical = np.zeros((count, 3, 8), np.float32)
            categorical = np.full((count, 3, 2), CATEGORICAL_PADDING_VALUE, np.int64)
            numerical[mask] = shard[f"{side}_object_numerical"][rows[mask]]
            categorical[mask] = shard[f"{side}_object_categorical"][rows[mask]]
            observation.update({f"{side}_object_numerical": numerical,
                                f"{side}_object_categorical": categorical, f"{side}_object_mask": mask})
        return observation

    def get_batch(self, indices):
        """按分片批量 gather；保持输入顺序及重复索引，不改变 PER 的抽样分布。"""
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("批量索引必须为非空一维数组")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("批量索引越界")
        shard_ids = self.index_shards[indices]
        result = None
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            frames = self.index_frames[indices[positions]].astype(np.int64)
            shard = self._load_shard(int(shard_id))
            size = len(frames)
            # 用 float64 按原先顺序累计，最后转 float32，与逐条 Python float 回报一致。
            rewards, discounts = np.zeros(size, np.float64), np.ones(size, np.float64)
            terminals, active = np.zeros(size, bool), np.ones(size, bool)
            final_frames = frames.copy()
            # PER 可能每个分片只抽一条；避免为这种情况创建 N 步小数组循环。
            if size == 1:
                rewards[0], discounts[0], terminals[0], final_frames[0] = self._n_step(shard, int(frames[0]))
            for offset in range(self.n_step if size > 1 else 0):
                rows = frames + offset
                active &= rows < len(shard["actions"]) - 1
                eligible = np.flatnonzero(active)
                active[eligible] &= shard["transition_valid"][rows[eligible]]
                eligible = np.flatnonzero(active)
                if not len(eligible):
                    break
                selected = rows[eligible]
                rewards[eligible] += discounts[eligible] * shard["rewards"][selected].astype(np.float64)
                discounts[eligible] *= self.gamma
                final_frames[eligible] = selected + 1
                terminals[eligible] = shard["dones"][selected]
                active[eligible] &= ~terminals[eligible]
            # 三种状态一起构造，终局导致的重复状态只 gather 一次；模型计算顺序不变。
            unique_frames, inverse = np.unique(np.concatenate((frames, frames + 1, final_frames)), return_inverse=True)
            observations = self._observations_many(shard, unique_frames)
            batch = {name: {key: value[inverse[i * size:(i + 1) * size]] for key, value in observations.items()}
                     for i, name in enumerate(("observation", "next_observation", "n_step_observation"))}
            batch.update(action=shard["actions"][frames].astype(np.int64),
                         reward=shard["rewards"][frames].astype(np.float32), done=shard["dones"][frames].astype(bool),
                         n_step_reward=rewards.astype(np.float32), n_step_discount=discounts.astype(np.float32),
                         n_step_done=terminals, is_demo=np.ones(size, bool),
                         training_weight=(shard["training_weights"][frames].astype(np.float32)
                                          if "training_weights" in shard else np.ones(size, np.float32)),
                         round_outcome=(shard["round_outcomes"][frames].astype(np.int8)
                                        if "round_outcomes" in shard else np.zeros(size, np.int8)))
            if self.evaluation_metadata:
                batch.update(active_weather=shard["state_categorical"][frames, 4].astype(np.int64),
                             object_count=sum(shard[f"{side}_object_offsets"][frames + 1] -
                                              shard[f"{side}_object_offsets"][frames] for side in ("self", "opponent")).astype(np.int64))
            if result is None:
                def allocate(value):
                    return {k: allocate(v) for k, v in value.items()} if isinstance(value, dict) else np.empty((len(indices), *value.shape[1:]), value.dtype)
                result = allocate(batch)
            def scatter(destination, source):
                for key, value in source.items():
                    if isinstance(value, dict):
                        scatter(destination[key], value)
                    else:
                        destination[key][positions] = value
            scatter(result, batch)
        def tensors(value):
            return {k: tensors(v) for k, v in value.items()} if isinstance(value, dict) else torch.from_numpy(value)
        return tensors(result)
