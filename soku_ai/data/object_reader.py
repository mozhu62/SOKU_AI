from __future__ import annotations

import numpy as np
import pandas as pd


ObjectFrameKey = tuple[int, int, str]


def build_object_group_index(objects: pd.DataFrame) -> dict[ObjectFrameKey, np.ndarray]:
    """建立 (sample_serial, battle_frame, side) 到对象行位置的稳定索引。"""
    grouped = objects.groupby(["sample_serial", "battle_frame", "side"], sort=False).indices
    return {
        (int(serial), int(frame), str(side)): np.asarray(indices, dtype=np.int64)
        for (serial, frame, side), indices in grouped.items()
    }
