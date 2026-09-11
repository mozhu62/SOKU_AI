from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .schemas import (
    HISTORY_NUMERICAL_FEATURES,
    OBJECT_NUMERICAL_FEATURES,
    STATE_CONTINUOUS_FEATURES,
)
from .sanitization import (
    DISABLED_OBJECT_NUMERICAL_FEATURES,
    sanitize_object_numerical,
)


@dataclass
class RunningMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.dimension, dtype=np.float64)
        self.sum_squares = np.zeros(self.dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        matrix = np.asarray(values, dtype=np.float64).reshape(-1, self.dimension)
        finite = np.isfinite(matrix)
        safe = np.where(finite, matrix, 0.0)
        self.sum += safe.sum(axis=0)
        self.sum_squares += np.square(safe).sum(axis=0)
        # 当前导出字段应整列有限；保留统一 count 可及时暴露异常数据
        if not finite.all():
            raise ValueError("拟合归一化参数时发现 NaN 或 Inf")
        self.count += matrix.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return np.zeros(self.dimension, dtype=np.float32), np.ones(self.dimension, dtype=np.float32)
        mean = self.sum / self.count
        variance = np.maximum(self.sum_squares / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    history_mean: np.ndarray
    history_std: np.ndarray
    object_mean: np.ndarray
    object_std: np.ndarray
    fitted_replays: tuple[str, ...]

    def normalize_state(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.state_mean) / self.state_std

    def normalize_history(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.history_mean) / self.history_std

    def normalize_objects(self, values: np.ndarray) -> np.ndarray:
        sanitized = sanitize_object_numerical(values)
        return (sanitized - self.object_mean) / self.object_std

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "state_features": list(STATE_CONTINUOUS_FEATURES),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "history_features": list(HISTORY_NUMERICAL_FEATURES),
            "history_mean": self.history_mean.tolist(),
            "history_std": self.history_std.tolist(),
            "object_features": list(OBJECT_NUMERICAL_FEATURES),
            "object_mean": self.object_mean.tolist(),
            "object_std": self.object_std.tolist(),
            "disabled_object_features": list(DISABLED_OBJECT_NUMERICAL_FEATURES),
            "fitted_replays": list(self.fitted_replays),
        }


def save_normalization(stats: NormalizationStats, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(stats.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")


def normalization_from_dict(payload: Mapping[str, object]) -> NormalizationStats:
    """从 checkpoint 内嵌数据恢复归一化参数，并校验字段顺序。"""
    if tuple(payload["state_features"]) != STATE_CONTINUOUS_FEATURES:
        raise ValueError("state normalization 字段顺序与当前 Schema 不一致")
    if tuple(payload["history_features"]) != HISTORY_NUMERICAL_FEATURES:
        raise ValueError("history normalization 字段顺序与当前 Schema 不一致")
    if tuple(payload["object_features"]) != OBJECT_NUMERICAL_FEATURES:
        raise ValueError("object normalization 字段顺序与当前 Schema 不一致")
    if tuple(payload.get("disabled_object_features", ())) != DISABLED_OBJECT_NUMERICAL_FEATURES:
        raise ValueError("object normalization 的禁用字段与当前配置不一致")
    return NormalizationStats(
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
        state_std=np.asarray(payload["state_std"], dtype=np.float32),
        history_mean=np.asarray(payload["history_mean"], dtype=np.float32),
        history_std=np.asarray(payload["history_std"], dtype=np.float32),
        object_mean=np.asarray(payload["object_mean"], dtype=np.float32),
        object_std=np.asarray(payload["object_std"], dtype=np.float32),
        fitted_replays=tuple(payload["fitted_replays"]),
    )


def load_normalization(path: str | Path) -> NormalizationStats:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return normalization_from_dict(payload)


def fit_normalization(
    shard_paths: Iterable[str | Path],
    replay_ids: Iterable[str],
) -> NormalizationStats:
    state = RunningMoments(len(STATE_CONTINUOUS_FEATURES))
    history = RunningMoments(len(HISTORY_NUMERICAL_FEATURES))
    objects = RunningMoments(len(OBJECT_NUMERICAL_FEATURES))
    fitted: list[str] = []
    for replay_id, shard_path in zip(replay_ids, shard_paths, strict=True):
        with np.load(shard_path, allow_pickle=False) as shard:
            state.update(shard["state_continuous"])
            history.update(shard["history_numerical"])
            objects.update(sanitize_object_numerical(shard["self_object_numerical"]))
            objects.update(sanitize_object_numerical(shard["opponent_object_numerical"]))
        fitted.append(replay_id)
    state_mean, state_std = state.finalize()
    history_mean, history_std = history.finalize()
    object_mean, object_std = objects.finalize()
    return NormalizationStats(
        state_mean,
        state_std,
        history_mean,
        history_std,
        object_mean,
        object_std,
        tuple(fitted),
    )
