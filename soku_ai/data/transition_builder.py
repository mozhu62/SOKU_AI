from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import NormalizationStats
from .schemas import (
    HISTORY_CATEGORICAL_FEATURES,
    HISTORY_NUMERICAL_FEATURES,
    OBJECT_CATEGORICAL_FEATURES,
    OBJECT_NUMERICAL_FEATURES,
    SCHEMA_VERSION,
)


CATEGORICAL_PADDING_VALUE = -2


class ProcessedReplayDataset(Dataset[dict[str, Any]]):
    """按索引读取预处理分片，并在取样时构造历史窗口和 N-step Transition。"""

    def __init__(
        self,
        processed_dir: str | Path,
        replay_ids: Iterable[str],
        normalization: NormalizationStats,
        *,
        history_len: int,
        max_objects_per_side: int,
        action_shift: int,
        gamma: float,
        n_step: int,
        shard_cache_size: int = 16,
        include_evaluation_metadata: bool = False,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.replay_ids = tuple(replay_ids)
        self.normalization = normalization
        self.history_len = int(history_len)
        self.max_objects = int(max_objects_per_side)
        self.action_shift = int(action_shift)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.include_evaluation_metadata = bool(include_evaluation_metadata)
        self.shard_paths = tuple(self.processed_dir / f"{replay_id}.npz" for replay_id in self.replay_ids)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        shard_indices: list[np.ndarray] = []
        frame_indices: list[np.ndarray] = []
        for shard_index, path in enumerate(self.shard_paths):
            if not path.is_file():
                raise FileNotFoundError(f"缺少预处理分片: {path}")
            with np.load(path, allow_pickle=False) as shard:
                metadata = json.loads(str(shard["metadata_json"].item()))
                if metadata.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError(f"分片 Schema 版本不一致: {path}")
                if int(metadata.get("action_shift", -1)) != self.action_shift:
                    raise ValueError(f"分片 action_shift 与当前配置不一致，请重新预处理: {path}")
                if int(metadata.get("max_objects_per_side", -1)) != self.max_objects:
                    raise ValueError(
                        f"分片 max_objects_per_side 与当前配置不一致，请重新预处理: {path}"
                    )
                valid = np.flatnonzero(shard["transition_valid"]).astype(np.int32)
            shard_indices.append(np.full(len(valid), shard_index, dtype=np.int16))
            frame_indices.append(valid)
        self.index_shards = (
            np.concatenate(shard_indices) if shard_indices else np.empty(0, dtype=np.int16)
        )
        self.index_frames = (
            np.concatenate(frame_indices) if frame_indices else np.empty(0, dtype=np.int32)
        )

    def __len__(self) -> int:
        return len(self.index_frames)

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        cached = self._cache.pop(shard_index, None)
        if cached is not None:
            self._cache[shard_index] = cached
            return cached
        with np.load(self.shard_paths[shard_index], allow_pickle=False) as file:
            shard = {name: file[name] for name in file.files}
        # 分片只加载一次，因此在缓存入口完成归一化，避免每个 Transition 重复计算。
        shard["state_continuous"] = self.normalization.normalize_state(
            shard["state_continuous"]
        )
        shard["history_numerical"] = self.normalization.normalize_history(
            shard["history_numerical"]
        )
        shard["self_object_numerical"] = self.normalization.normalize_objects(
            shard["self_object_numerical"]
        )
        shard["opponent_object_numerical"] = self.normalization.normalize_objects(
            shard["opponent_object_numerical"]
        )
        self._cache[shard_index] = shard
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return shard

    def preload_shards(self) -> None:
        """训练前一次性载入全部分片，避免 PER 随机采样反复解包 NPZ。"""
        if self.shard_cache_size < len(self.shard_paths):
            raise ValueError(
                "预加载要求 shard_cache_size 不小于分片数量: "
                f"cache={self.shard_cache_size}, shards={len(self.shard_paths)}"
            )
        for shard_index in range(len(self.shard_paths)):
            self._load_shard(shard_index)

    def _history(self, shard: dict[str, np.ndarray], frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # shift=0 时排除当前帧输入，防止 expert action 泄漏到 history
        end = frame_index + (1 if self.action_shift > 0 else 0)
        episode_start = int(shard["episode_start"][frame_index])
        start = max(episode_start, end - self.history_len)
        count = max(0, end - start)
        numerical = np.zeros(
            (self.history_len, len(HISTORY_NUMERICAL_FEATURES)), dtype=np.float32
        )
        categorical = np.full(
            (self.history_len, len(HISTORY_CATEGORICAL_FEATURES)),
            CATEGORICAL_PADDING_VALUE,
            dtype=np.int64,
        )
        mask = np.zeros(self.history_len, dtype=np.bool_)
        if count:
            destination = slice(self.history_len - count, self.history_len)
            numerical[destination] = shard["history_numerical"][start:end]
            categorical[destination] = shard["history_categorical"][start:end]
            mask[destination] = True
        return numerical, categorical, mask

    def _objects(
        self,
        shard: dict[str, np.ndarray],
        frame_index: int,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = shard[f"{side}_object_offsets"]
        start = int(offsets[frame_index])
        end = int(offsets[frame_index + 1])
        count = min(end - start, self.max_objects)
        numerical = np.zeros(
            (self.max_objects, len(OBJECT_NUMERICAL_FEATURES)), dtype=np.float32
        )
        categorical = np.full(
            (self.max_objects, len(OBJECT_CATEGORICAL_FEATURES)),
            CATEGORICAL_PADDING_VALUE,
            dtype=np.int64,
        )
        mask = np.zeros(self.max_objects, dtype=np.bool_)
        if count:
            numerical[:count] = shard[f"{side}_object_numerical"][start : start + count]
            categorical[:count] = shard[f"{side}_object_categorical"][start : start + count]
            mask[:count] = True
        return numerical, categorical, mask

    def _observation(self, shard: dict[str, np.ndarray], frame_index: int) -> dict[str, torch.Tensor]:
        history_num, history_cat, history_mask = self._history(shard, frame_index)
        self_obj_num, self_obj_cat, self_obj_mask = self._objects(shard, frame_index, "self")
        opponent_obj_num, opponent_obj_cat, opponent_obj_mask = self._objects(
            shard, frame_index, "opponent"
        )
        return {
            "state_continuous": torch.from_numpy(shard["state_continuous"][frame_index]),
            "state_categorical": torch.from_numpy(shard["state_categorical"][frame_index]),
            "history_numerical": torch.from_numpy(history_num),
            "history_categorical": torch.from_numpy(history_cat),
            "history_mask": torch.from_numpy(history_mask),
            "self_object_numerical": torch.from_numpy(self_obj_num),
            "self_object_categorical": torch.from_numpy(self_obj_cat),
            "self_object_mask": torch.from_numpy(self_obj_mask),
            "opponent_object_numerical": torch.from_numpy(opponent_obj_num),
            "opponent_object_categorical": torch.from_numpy(opponent_obj_cat),
            "opponent_object_mask": torch.from_numpy(opponent_obj_mask),
        }

    def _n_step(self, shard: dict[str, np.ndarray], start_index: int) -> tuple[float, float, bool, int]:
        reward = 0.0
        discount = 1.0
        terminal = False
        final_state = start_index
        episode_end = int(shard["episode_end"][start_index])
        for offset in range(self.n_step):
            transition_index = start_index + offset
            if transition_index >= episode_end - 1 or not shard["transition_valid"][transition_index]:
                break
            reward += discount * float(shard["rewards"][transition_index])
            final_state = transition_index + 1
            terminal = bool(shard["dones"][transition_index])
            discount *= self.gamma
            if terminal:
                break
        return reward, discount, terminal, final_state

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index = int(self.index_shards[index])
        frame_index = int(self.index_frames[index])
        shard = self._load_shard(shard_index)
        n_reward, n_discount, n_done, n_state_index = self._n_step(shard, frame_index)
        training_weight = (
            float(shard["training_weights"][frame_index])
            if "training_weights" in shard
            else 1.0
        )
        round_outcome = (
            int(shard["round_outcomes"][frame_index])
            if "round_outcomes" in shard
            else 0
        )
        item = {
            "observation": self._observation(shard, frame_index),
            "next_observation": self._observation(shard, frame_index + 1),
            "n_step_observation": self._observation(shard, n_state_index),
            "action": torch.tensor(int(shard["actions"][frame_index]), dtype=torch.long),
            "reward": torch.tensor(float(shard["rewards"][frame_index]), dtype=torch.float32),
            "done": torch.tensor(bool(shard["dones"][frame_index]), dtype=torch.bool),
            "n_step_reward": torch.tensor(n_reward, dtype=torch.float32),
            "n_step_discount": torch.tensor(n_discount, dtype=torch.float32),
            "n_step_done": torch.tensor(n_done, dtype=torch.bool),
            "is_demo": torch.tensor(True, dtype=torch.bool),
            "training_weight": torch.tensor(training_weight, dtype=torch.float32),
            "round_outcome": torch.tensor(round_outcome, dtype=torch.int8),
        }
        if self.include_evaluation_metadata:
            state_cat = shard["state_categorical"][frame_index]
            self_offsets = shard["self_object_offsets"]
            opponent_offsets = shard["opponent_object_offsets"]
            item.update(
                {
                    "opponent_character_id": torch.tensor(
                        int(state_cat[2]), dtype=torch.long
                    ),
                    "displayed_weather": torch.tensor(
                        int(state_cat[6]), dtype=torch.long
                    ),
                    "object_count": torch.tensor(
                        int(self_offsets[frame_index + 1] - self_offsets[frame_index])
                        + int(
                            opponent_offsets[frame_index + 1]
                            - opponent_offsets[frame_index]
                        ),
                        dtype=torch.long,
                    ),
                }
            )
        return item
