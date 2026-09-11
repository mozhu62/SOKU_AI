from __future__ import annotations

import numpy as np

from .schemas import OBJECT_NUMERICAL_FEATURES


# 采集端 ObjectManager.gravity 对大量对象是未初始化内存，既包含 NaN，
# 也包含接近 float32 极限的伪数值，无法从现有 Replay 中可靠恢复。
DISABLED_OBJECT_NUMERICAL_FEATURES: tuple[str, ...] = (
    "gravity_x",
    "gravity_y",
)

_DISABLED_OBJECT_INDICES = tuple(
    OBJECT_NUMERICAL_FEATURES.index(name)
    for name in DISABLED_OBJECT_NUMERICAL_FEATURES
)
_ENABLED_OBJECT_INDICES = tuple(
    index
    for index in range(len(OBJECT_NUMERICAL_FEATURES))
    if index not in _DISABLED_OBJECT_INDICES
)


def sanitize_object_numerical(values: np.ndarray) -> np.ndarray:
    """禁用不可信的重力字段，同时保持其他对象字段的严格有限值约束。"""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != len(OBJECT_NUMERICAL_FEATURES):
        raise ValueError(
            "对象数值特征维度错误: "
            f"expected={len(OBJECT_NUMERICAL_FEATURES)}, actual={array.shape}"
        )

    enabled = array[..., _ENABLED_OBJECT_INDICES]
    if not np.isfinite(enabled).all():
        invalid = ~np.isfinite(array)
        invalid[..., _DISABLED_OBJECT_INDICES] = False
        bad_indices = np.flatnonzero(invalid.reshape(-1, invalid.shape[-1]).any(axis=0))
        bad_names = [OBJECT_NUMERICAL_FEATURES[int(index)] for index in bad_indices]
        raise ValueError(f"对象有效特征中发现 NaN 或 Inf: {bad_names}")

    sanitized = array.copy()
    sanitized[..., _DISABLED_OBJECT_INDICES] = 0.0
    return sanitized
