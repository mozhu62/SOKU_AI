from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import OBJECT_SOURCE_FIELDS, required_main_columns


@dataclass(frozen=True)
class ReplayPair:
    replay_id: str
    main_path: Path
    objects_path: Path
    done_path: Path


def discover_replay_pairs(raw_dir: str | Path, require_done: bool = True) -> list[ReplayPair]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {root}")

    pairs: list[ReplayPair] = []
    for main_path in sorted(root.rglob("*.csv")):
        if main_path.name.endswith(".objects.csv"):
            continue
        objects_path = main_path.with_suffix(".objects.csv")
        done_path = Path(f"{main_path}.done")
        if not objects_path.is_file():
            raise FileNotFoundError(f"缺少对象表: {objects_path}")
        if require_done and not done_path.is_file():
            raise FileNotFoundError(f"缺少完成标记: {done_path}")
        replay_id = main_path.relative_to(root).with_suffix("").as_posix()
        pairs.append(ReplayPair(replay_id, main_path, objects_path, done_path))
    return pairs


def read_csv_metadata(path: str | Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].strip().partition("=")
            if separator:
                metadata[key.strip()] = value.strip()
    return metadata


def read_main_replay(pair: ReplayPair, extra_columns: Iterable[str] = ()) -> pd.DataFrame:
    columns = list(dict.fromkeys((*required_main_columns(), *extra_columns)))
    frame = pd.read_csv(pair.main_path, comment="#", usecols=columns, low_memory=False)
    frame.insert(0, "source_row", range(len(frame)))
    return frame


def read_objects(pair: ReplayPair, extra_columns: Iterable[str] = ()) -> pd.DataFrame:
    columns = list(dict.fromkeys((*OBJECT_SOURCE_FIELDS, *extra_columns)))
    return pd.read_csv(pair.objects_path, comment="#", usecols=columns, low_memory=False)


def read_done_counts(pair: ReplayPair) -> dict[str, int]:
    result: dict[str, int] = {}
    with pair.done_path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line in file:
            key, separator, value = line.strip().partition("=")
            if separator:
                result[key] = int(value)
    return result


def replay_contains_character(pair: ReplayPair, character_id: int) -> bool:
    first = pd.read_csv(
        pair.main_path,
        comment="#",
        usecols=["left_character_id", "right_character_id"],
        nrows=1,
    )
    if first.empty:
        return False
    return bool(
        int(first.iloc[0]["left_character_id"]) == int(character_id)
        or int(first.iloc[0]["right_character_id"]) == int(character_id)
    )


def filter_character_replays(
    pairs: Iterable[ReplayPair], character_id: int
) -> list[ReplayPair]:
    return [pair for pair in pairs if replay_contains_character(pair, character_id)]
