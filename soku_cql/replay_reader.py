from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayPair:
    replay_id: str
    main_path: Path
    objects_path: Path
    done_path: Path


def discover_replay_pairs(root: Path) -> list[ReplayPair]:
    if not root.is_dir():
        raise FileNotFoundError(f"Replay CSV 目录不存在：{root}")
    result = []
    for main in sorted(root.rglob("*.csv")):
        if main.name.endswith(".objects.csv"):
            continue
        objects, done = main.with_suffix(".objects.csv"), Path(f"{main}.done")
        if not objects.is_file() or not done.is_file():
            raise FileNotFoundError(f"Replay 数据不完整：{main}")
        result.append(ReplayPair(main.relative_to(root).with_suffix("").as_posix(), main, objects, done))
    return result


def object_group_index(objects: pd.DataFrame) -> dict[tuple[int, int, str], np.ndarray]:
    grouped = objects.groupby(["sample_serial", "battle_frame", "side"], sort=False).indices
    return {(int(serial), int(frame), str(side)): np.asarray(rows, dtype=np.int64)
            for (serial, frame, side), rows in grouped.items()}

