from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .replay_reader import ReplayPair, read_done_counts
from .schemas import required_main_columns


@dataclass
class ReplayValidationResult:
    replay_id: str
    valid: bool = True
    frame_count: int = 0
    object_row_count: int = 0
    valid_battle_frames: int = 0
    round_segments: int = 0
    sample_serial_gaps: int = 0
    battle_frame_gaps: int = 0
    orphan_object_keys: int = 0
    object_count_mismatches: int = 0
    total_count_mismatches: int = 0
    object_reported_total_mismatches: int = 0
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_replay_pair(
    pair: ReplayPair,
    *,
    valid_match_states: Iterable[int] = (2,),
    valid_battle_sub_modes: Iterable[int] = (2,),
) -> ReplayValidationResult:
    result = ReplayValidationResult(replay_id=pair.replay_id)
    main_columns = list(required_main_columns())
    main = pd.read_csv(pair.main_path, comment="#", usecols=main_columns, low_memory=False)
    result.frame_count = len(main)
    done = read_done_counts(pair)
    if done.get("frames") != len(main):
        result.fail(f"完成标记帧数 {done.get('frames')} 与主表 {len(main)} 不一致")

    serial_diff = main["sample_serial"].astype(np.int64).diff().iloc[1:]
    result.sample_serial_gaps = int((serial_diff != 1).sum())
    if result.sample_serial_gaps:
        result.fail(f"sample_serial 非连续位置数量: {result.sample_serial_gaps}")

    valid = main[
        main["match_state"].isin(tuple(valid_match_states))
        & main["battle_sub_mode"].isin(tuple(valid_battle_sub_modes))
    ].copy()
    result.valid_battle_frames = len(valid)
    if valid.empty:
        result.fail("没有找到有效战斗帧")
    else:
        source_rows = valid.index.to_numpy(np.int64)
        frames = valid["battle_frame"].to_numpy(np.int64)
        rounds = valid["current_round"].to_numpy(np.int64)
        new_segment = np.ones(len(valid), dtype=np.bool_)
        if len(valid) > 1:
            new_segment[1:] = (
                (source_rows[1:] != source_rows[:-1] + 1)
                | (frames[1:] != frames[:-1] + 1)
                | (rounds[1:] != rounds[:-1])
            )
        result.round_segments = int(new_segment.sum())
        internal = (
            (source_rows[1:] == source_rows[:-1] + 1)
            & (rounds[1:] == rounds[:-1])
            & (frames[1:] != frames[:-1] + 1)
        )
        result.battle_frame_gaps = int(internal.sum())
        if result.battle_frame_gaps:
            result.fail(f"有效回合内部 battle_frame 缺口: {result.battle_frame_gaps}")

    object_columns = ["sample_serial", "battle_frame", "side", "total_object_count"]
    objects = pd.read_csv(pair.objects_path, comment="#", usecols=object_columns, low_memory=False)
    result.object_row_count = len(objects)
    if done.get("object_rows") != len(objects):
        result.fail(f"完成标记对象行数 {done.get('object_rows')} 与对象表 {len(objects)} 不一致")

    main_keys = pd.MultiIndex.from_frame(main[["sample_serial", "battle_frame"]])
    object_keys = pd.MultiIndex.from_frame(objects[["sample_serial", "battle_frame"]])
    result.orphan_object_keys = int((~object_keys.isin(main_keys)).sum())
    if result.orphan_object_keys:
        result.fail(f"对象表中无法匹配主表的行数: {result.orphan_object_keys}")

    actual = (
        objects.groupby(["sample_serial", "battle_frame", "side"], sort=False)
        .size()
        .unstack("side", fill_value=0)
    )
    for side in ("left", "right"):
        if side not in actual.columns:
            actual[side] = 0
    actual = actual.reindex(main_keys, fill_value=0)
    reported_total = (
        objects.groupby(["sample_serial", "battle_frame", "side"], sort=False)["total_object_count"]
        .max()
        .unstack("side", fill_value=0)
    )
    for side in ("left", "right"):
        if side not in reported_total.columns:
            reported_total[side] = 0
    reported_total = reported_total.reindex(main_keys, fill_value=0)
    for side in ("left", "right"):
        recorded = main[f"{side}_recorded_object_count"].to_numpy(np.int64)
        total = main[f"{side}_total_object_count"].to_numpy(np.int64)
        actual_side = actual[side].to_numpy(np.int64)
        mismatch = actual_side != recorded
        result.object_count_mismatches += int(mismatch.sum())
        result.total_count_mismatches += int((total < recorded).sum())
        has_objects = actual_side > 0
        result.object_reported_total_mismatches += int(
            (has_objects & (reported_total[side].to_numpy(np.int64) != total)).sum()
        )
    if result.object_count_mismatches:
        result.fail(f"主表 recorded_object_count 与对象行数不一致: {result.object_count_mismatches}")
    if result.total_count_mismatches:
        result.fail(f"total_object_count 小于 recorded_object_count: {result.total_count_mismatches}")
    if result.object_reported_total_mismatches:
        result.fail(
            "对象表 total_object_count 与主表不一致: "
            f"{result.object_reported_total_mismatches}"
        )
    return result


def save_validation_report(results: list[ReplayValidationResult], path: str | Path) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "valid": all(result.valid for result in results),
        "replays": [result.to_dict() for result in results],
    }
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
