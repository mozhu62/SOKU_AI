from __future__ import annotations

import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .action_space import ACTION_SCHEMA, CARD_OVERLAP_POLICY, BUTTON_COLUMNS, compact_actions, clean_card_overlap
from .resources import RESOURCE_SUFFIXES, validate_player_resources
from .replay_reader import ReplayPair, discover_replay_pairs, object_group_index
from .schema import (
    CQL_REPLAY_SCHEMA, MAX_HAND_CARDS, OBJECT_COLUMNS, SKILL_COMMANDS, STATE_CATEGORICAL_FEATURES,
    STATE_CONTINUOUS_FEATURES, OPTIONAL_STATE_FEATURES, UNKNOWN_CARD_ID, manifest, required_main_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUARDING_FLAG = 1 << 11
GRAZE_FLAG = 1 << 10
AIRBORNE_FLAG = 1 << 2


@dataclass(frozen=True)
class ConversionResult:
    replay_id: str
    frames: int
    transitions: int
    output: str
    card_overlap_cleaned_rows: int = 0


def _save_npz(destination: Path, compression_level: int, **arrays: np.ndarray) -> None:
    """使用较低压缩等级写入标准 NPZ，减少大批量转换时的 CPU 等待。"""
    if not 0 <= compression_level <= 9:
        raise ValueError("npz_compression_level 必须位于 0 到 9")
    compression = zipfile.ZIP_STORED if compression_level == 0 else zipfile.ZIP_DEFLATED
    level = None if compression_level == 0 else compression_level
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=compression,
        compresslevel=level,
        allowZip64=True,
    ) as archive:
        for name, value in arrays.items():
            # NPZ 本质上是多个 .npy 文件组成的 ZIP；流式写入避免额外复制整个分片。
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _is_current_shard(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            if metadata.get("dataset_schema") != CQL_REPLAY_SCHEMA:
                return False
            buttons = source["action_buttons"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    # 已有原始 NPZ 在训练加载时清洗，不为这个规则重写文件、改变固定划分哈希。
    _, overlap = clean_card_overlap(buttons, str(path))
    if np.any(overlap):
        print(f"[clean] {path.name} 含 {int(np.count_nonzero(overlap))} 行卡键重合，加载时保留用卡、清除切卡")
    return True


def _values(frame: pd.DataFrame, side: str, field: str, dtype=np.float32) -> np.ndarray:
    return frame[f"{side}_{field}"].to_numpy(dtype=dtype, copy=False)


def _sides(frame: pd.DataFrame, character_id: int) -> tuple[str, str]:
    left = _values(frame, "left", "character_id", np.int64) == character_id
    right = _values(frame, "right", "character_id", np.int64) == character_id
    if left.all() and not right.any():
        return "left", "right"
    if right.all() and not left.any():
        return "right", "left"
    raise ValueError("Replay 中必须恰好有一侧始终是萃香")


def _episodes(frame: pd.DataFrame) -> np.ndarray:
    source = frame["source_row"].to_numpy(np.int64)
    battle = frame["battle_frame"].to_numpy(np.int64)
    rounds = frame["current_round"].to_numpy(np.int64)
    starts = np.ones(len(frame), dtype=np.bool_)
    if len(frame) > 1:
        starts[1:] = ((source[1:] != source[:-1] + 1) | (battle[1:] != battle[:-1] + 1)
                      | (rounds[1:] != rounds[:-1]))
    episode = np.cumsum(starts, dtype=np.int32) - 1
    return episode


def _state(frame, self_side, opponent_side):
    sx, sy = _values(frame, self_side, "position_x"), _values(frame, self_side, "position_y")
    ox, oy = _values(frame, opponent_side, "position_x"), _values(frame, opponent_side, "position_y")
    columns = {
        "self_position_x": sx, "self_position_y": sy,
        "self_speed_x": _values(frame, self_side, "speed_x"),
        "self_speed_y": _values(frame, self_side, "speed_y"),
        "self_direction": _values(frame, self_side, "direction"),
        "self_current_spirit": _values(frame, self_side, "current_spirit"),
        "self_action_frame_count": _values(frame, self_side, "action_frame_count"),
        "opponent_position_x": ox, "opponent_position_y": oy,
        "opponent_speed_x": _values(frame, opponent_side, "speed_x"),
        "opponent_speed_y": _values(frame, opponent_side, "speed_y"),
        "opponent_direction": _values(frame, opponent_side, "direction"),
        "opponent_current_spirit": _values(frame, opponent_side, "current_spirit"),
        "opponent_action_frame_count": _values(frame, opponent_side, "action_frame_count"),
        "relative_x": ox - sx, "relative_y": oy - sy,
    }
    columns["self_hp"] = np.maximum(_values(frame, self_side, "hp"), 0)
    columns["opponent_hp"] = np.maximum(_values(frame, opponent_side, "hp"), 0)
    # 转换阶段保留原值；CQL 训练前在固定训练集上单独拟合归一化。
    continuous = np.column_stack([columns[name] for name in STATE_CONTINUOUS_FEATURES]).astype(np.float32)
    categories = {
        "self_action": _values(frame, self_side, "action", np.int64),
        "self_action_block_id": _values(frame, self_side, "action_block_id", np.int64),
        "opponent_action": _values(frame, opponent_side, "action", np.int64),
        "opponent_action_block_id": _values(frame, opponent_side, "action_block_id", np.int64),
        "active_weather": frame["active_weather"].to_numpy(np.int64),
    }
    categorical = np.column_stack([categories[name] for name in STATE_CATEGORICAL_FEATURES]).astype(np.int64)
    if not np.isfinite(continuous).all():
        raise ValueError("角色输入中发现 NaN 或 Inf")
    return continuous, categorical


def _resources(frame: pd.DataFrame, source_side: str, output_side: str) -> dict[str, np.ndarray]:
    """保留玩家在当帧可见的技能配置和卡槽；不把未抽取卡组顺序作为模型可见状态。"""
    skill = lambda field: np.column_stack([
        _values(frame, source_side, f"skill_{command}_{field}", np.int8)
        for command in SKILL_COMMANDS
    ]).astype(np.int8)
    hand_ids = np.column_stack([
        _values(frame, source_side, f"hand_{index}_id", np.uint16)
        for index in range(MAX_HAND_CARDS)
    ])
    hand_costs = np.column_stack([
        _values(frame, source_side, f"hand_{index}_cost", np.uint16)
        for index in range(MAX_HAND_CARDS)
    ])
    hand_count = np.clip(
        _values(frame, source_side, "hand_count", np.int16), 0, MAX_HAND_CARDS
    )
    hand_mask = ((np.arange(MAX_HAND_CARDS)[None, :] < hand_count[:, None]) & (hand_ids != UNKNOWN_CARD_ID))
    card_state = np.column_stack((
        _values(frame, source_side, "card_gauge", np.int32),
        _values(frame, source_side, "card_count", np.int32),
        _values(frame, source_side, "selected_card_index", np.int32),
        _values(frame, source_side, "selected_card_id", np.int32),
        _values(frame, source_side, "selected_card_cost", np.int32),
        _values(frame, source_side, "hand_capacity", np.int32),
        hand_count.astype(np.int32),
        _values(frame, source_side, "hand_cards_used", np.int32),
    )).astype(np.int32)
    return {
        f"{output_side}_skill_valid_mask": _values(
            frame, source_side, "skill_valid_mask", np.uint8
        ),
        f"{output_side}_skill_variants": skill("variant"),
        f"{output_side}_skill_levels": skill("level"),
        f"{output_side}_skill_effective_levels": skill("effective_level"),
        f"{output_side}_card_state": card_state,
        f"{output_side}_hand_card_ids": hand_ids,
        f"{output_side}_hand_card_costs": hand_costs,
        f"{output_side}_hand_mask": hand_mask.astype(np.uint8),
    }


def _pack_objects(frame, objects, groups, source_side, self_side, maximum):
    sx, sy = _values(frame, self_side, "position_x"), _values(frame, self_side, "position_y")
    svx, svy = _values(frame, self_side, "speed_x"), _values(frame, self_side, "speed_y")
    serials = frame["sample_serial"].to_numpy(np.int64)
    battles = frame["battle_frame"].to_numpy(np.int64)
    numerical_parts, categorical_parts = [], []
    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    projectile_active = np.zeros(len(frame), dtype=np.float32)
    truncated = 0
    for frame_index, (serial, battle) in enumerate(zip(serials, battles, strict=True)):
        indices = groups.get((int(serial), int(battle), source_side))
        if indices is None or not len(indices):
            offsets[frame_index + 1] = offsets[frame_index]
            continue
        entity = objects.iloc[indices]
        dx = entity["position_x"].to_numpy(np.float32) - sx[frame_index]
        dy = entity["position_y"].to_numpy(np.float32) - sy[frame_index]
        # 只保留距离萃香最近的对象；相同距离按原始序号稳定排序。
        order = np.lexsort((entity["object_index"].to_numpy(np.int64), np.hypot(dx, dy)))
        truncated += max(0, len(order) - maximum)
        order = order[:maximum]
        selected = entity.iloc[order]
        dx, dy = dx[order], dy[order]
        raw = np.column_stack((
            dx, dy,
            selected["speed_x"].to_numpy(np.float32) - svx[frame_index],
            selected["speed_y"].to_numpy(np.float32) - svy[frame_index],
            selected["direction"].to_numpy(np.float32),
            selected["action_frame_count"].to_numpy(np.float32),
            selected["hitstop"].to_numpy(np.float32),
            selected["hit_count"].to_numpy(np.float32),
        ))
        if not np.isfinite(raw).all():
            raise ValueError(f"对象有效输入出现 NaN/Inf：frame={int(battle)} side={source_side}")
        numerical_parts.append(raw.astype(np.float32))
        categorical_parts.append(np.column_stack((
            selected["action"].to_numpy(np.int64), selected["action_block_id"].to_numpy(np.int64),
        )))
        offsets[frame_index + 1] = offsets[frame_index] + len(selected)
        available = selected["frame_data_available"].to_numpy(np.int64) != 0
        hitbox = np.zeros(len(selected), dtype=np.bool_)
        for box in range(5):
            hitbox |= ((selected[f"hit_box_{box}_valid"].to_numpy(np.int64) != 0)
                       & (selected[f"hit_rotation_box_{box}_valid"].to_numpy(np.int64) == 0))
        projectile_active[frame_index] = float(np.any(available & hitbox))
    numerical = np.concatenate(numerical_parts) if numerical_parts else np.empty((0, 8), np.float32)
    categorical = np.concatenate(categorical_parts) if categorical_parts else np.empty((0, 2), np.int64)
    return numerical, categorical, offsets, projectile_active, truncated


def _tactical(frame, self_side, opponent_side, opponent_projectiles):
    self_flags = _values(frame, self_side, "frame_flags", np.int64)
    opponent_flags = _values(frame, opponent_side, "frame_flags", np.int64)
    self_available = _values(frame, self_side, "frame_data_available", np.int64) != 0
    opponent_available = _values(frame, opponent_side, "frame_data_available", np.int64) != 0
    self_action = _values(frame, self_side, "action", np.int64)
    opponent_action = _values(frame, opponent_side, "action", np.int64)
    return np.column_stack((
        self_available & ((self_flags & GUARDING_FLAG) != 0),
        opponent_available & ((opponent_flags & GUARDING_FLAG) != 0),
        self_available & ((self_flags & GRAZE_FLAG) != 0),
        opponent_available & ((opponent_flags & GRAZE_FLAG) != 0),
        opponent_projectiles,
        (self_action >= 50) & (self_action < 150),
        self_available & ((self_flags & AIRBORNE_FLAG) != 0),
        (opponent_action >= 50) & (opponent_action < 150),
        opponent_available & ((opponent_flags & AIRBORNE_FLAG) != 0),
    )).astype(np.float32)


def convert_replay(pair: ReplayPair, destination: Path, config: dict) -> ConversionResult:
    data = config["data"]
    optional_columns = {f"{side}_{field}" for side in ("left", "right") for field in ("max_spirit", "hitstop")}
    required = set(required_main_columns())
    frame = pd.read_csv(pair.main_path, comment="#", usecols=lambda key: key in required | optional_columns,
                        low_memory=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Replay CSV 缺少资源字段：{sorted(missing)}；不能凭空补造，需有对应的新采集数据")
    # 保留原始 CSV 的计数值；这里只记录重合位置，最终输出按钮时再清洗。
    card_overlap_masks = {}
    for side in ("left", "right"):
        _, card_overlap_masks[side] = clean_card_overlap(
            np.column_stack([frame[f"{side}_{key}"].to_numpy() > 0 for key in BUTTON_COLUMNS]),
            f"{pair.replay_id} {side} 原始 CSV")
    frame = pd.concat((pd.DataFrame({"source_row": np.arange(len(frame))}), frame), axis=1)
    self_side, opponent_side = _sides(frame, int(data["suika_character_id"]))
    valid = ((frame["initialized"] != 0) & (frame["in_battle"] != 0)
             & frame["match_state"].isin(data["valid_match_states"])
             & frame["battle_sub_mode"].isin(data["valid_battle_sub_modes"]))
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("有效战斗帧少于 2")
    episode = _episodes(frame)
    continuous, categorical = _state(frame, self_side, opponent_side)
    optional_values = np.zeros((len(frame), 4), np.float32)
    optional_mask = np.zeros((len(frame), 4), bool)
    for index, (source_side, field) in enumerate((source_side, field)
            for source_side in (self_side, opponent_side) for field in ("max_spirit", "hitstop")):
        column = f"{source_side}_{field}"
        if column in frame:
            values = frame[column].to_numpy(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{column} 含 NaN/Inf")
            optional_values[:, index], optional_mask[:, index] = values, True
    resources = {
        **_resources(frame, self_side, "self"),
        **_resources(frame, opponent_side, "opponent"),
    }
    for side in ("self", "opponent"):
        validate_player_resources({key: resources[f"{side}_{key}"] for key in RESOURCE_SUFFIXES}, (len(frame),), side)
    objects = pd.read_csv(pair.objects_path, comment="#", usecols=list(OBJECT_COLUMNS), low_memory=False)
    groups = object_group_index(objects)
    maximum = int(data.get("nearest_projectiles_per_side", 3))
    if maximum != 3:
        raise ValueError("nearest_projectiles_per_side 当前必须为 3")
    self_num, self_cat, self_offsets, _, self_truncated = _pack_objects(
        frame, objects, groups, self_side, self_side, maximum)
    opponent_num, opponent_cat, opponent_offsets, projectile_active, opponent_truncated = _pack_objects(
        frame, objects, groups, opponent_side, self_side, maximum)
    tactical = _tactical(frame, self_side, opponent_side, projectile_active)

    transition_valid = np.zeros(len(frame), dtype=np.bool_)
    transition_valid[:-1] = episode[:-1] == episode[1:]
    action_shift = int(data.get("action_shift", 1))
    if action_shift not in (0, 1):
        raise ValueError("action_shift 当前只允许 0 或 1")
    action_source = np.minimum(np.arange(len(frame)) + action_shift, len(frame) - 1)
    action_horizontal, action_vertical, action_duration, action_buttons = compact_actions(
        frame, self_side, action_source, episode
    )
    # 统计与 NPZ 标签同一套过滤/偏移后的行，避免把原始 CSV 行数当作训练标签数。
    card_overlap_cleaned_rows = int(np.count_nonzero(
        card_overlap_masks[self_side][frame["source_row"].to_numpy(np.int64)][action_source]))
    action_valid = transition_valid & (episode[action_source] == episode)
    self_hp = _values(frame, self_side, "hp", np.int32)
    opponent_hp = _values(frame, opponent_side, "hp", np.int32)
    dealt = np.zeros(len(frame), dtype=np.float32)
    taken = np.zeros(len(frame), dtype=np.float32)
    dealt[:-1] = np.maximum(0, opponent_hp[:-1] - np.maximum(0, opponent_hp[1:]))
    taken[:-1] = np.maximum(0, self_hp[:-1] - np.maximum(0, self_hp[1:]))
    dealt[~transition_valid], taken[~transition_valid] = 0, 0
    reward = dealt * float(config["reward"]["damage_dealt"]) - taken * float(config["reward"]["damage_taken"])
    terminated = np.zeros(len(frame), dtype=np.bool_)
    terminated[:-1] = transition_valid[:-1] & ((self_hp[1:] <= 0) | (opponent_hp[1:] <= 0))
    metadata = {
        **manifest(), "replay_id": pair.replay_id, "suika_side": self_side,
        "nearest_projectiles_per_side": maximum,
        "self_projectiles_discarded": int(self_truncated),
        "opponent_projectiles_discarded": int(opponent_truncated),
        "action_shift": action_shift,
        "continuous_normalized": False,
        "action_schema": ACTION_SCHEMA,
        "card_overlap_policy": CARD_OVERLAP_POLICY,
        "card_overlap_cleaned_rows": card_overlap_cleaned_rows,
        "action_semantics": "single_game_frame_controller_state; labels decoded as joint432",
        "action_duration_semantics": "consecutive_horizontal_vertical_combination_frames_not_joint_action_duration",
        "optional_state_continuous": list(OPTIONAL_STATE_FEATURES),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(
        destination, int(data.get("npz_compression_level", 1)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        episode_id=episode,
        state_continuous=continuous, state_categorical=categorical,
        state_optional_continuous=optional_values, state_optional_mask=optional_mask,
        tactical_state=tactical, self_object_numerical=self_num,
        self_object_categorical=self_cat, self_object_offsets=self_offsets,
        opponent_object_numerical=opponent_num, opponent_object_categorical=opponent_cat,
        opponent_object_offsets=opponent_offsets,
        action_horizontal=action_horizontal, action_vertical=action_vertical,
        action_duration=action_duration, action_buttons=action_buttons,
        **resources,
        transition_valid=(transition_valid & action_valid),
        rewards=reward.astype(np.float32), terminated=terminated,
    )
    return ConversionResult(pair.replay_id, len(frame), int(transition_valid.sum()), str(destination),
                            card_overlap_cleaned_rows)


def _convert_job(pair: ReplayPair, destination: Path, config: dict) -> ConversionResult:
    return convert_replay(pair, destination, config)


def convert_dataset(
    config_path: str,
    overwrite: bool,
    limit: int | None,
    workers: int | None = None,
) -> None:
    config_file = _resolve(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    pairs = discover_replay_pairs(_resolve(config["data"]["raw_dir"]))
    if not pairs:
        raise FileNotFoundError(
            f"采集目录中没有完整的 CSV 三件套：{_resolve(config['data']['raw_dir'])}"
        )
    if limit is not None:
        pairs = pairs[:max(0, limit)]
    output = _resolve(config["data"]["output_dir"])
    pending: list[tuple[int, ReplayPair, Path]] = []
    for index, pair in enumerate(pairs, 1):
        destination = output / f"{pair.replay_id}.npz"
        if destination.exists() and not overwrite and _is_current_shard(destination):
            print(f"[{index}/{len(pairs)}] Skip {pair.replay_id}")
            continue
        pending.append((index, pair, destination))

    worker_count = int(workers if workers is not None else config["data"].get("conversion_workers", 1))
    if worker_count < 1:
        raise ValueError("conversion_workers 必须大于等于 1")
    results: list[ConversionResult] = []
    if worker_count == 1:
        for index, pair, destination in pending:
            result = convert_replay(pair, destination, config)
            results.append(result)
            print(f"[{index}/{len(pairs)}] {result.replay_id}: frames={result.frames} "
                  f"transitions={result.transitions} card_overlap_cleaned={result.card_overlap_cleaned_rows} "
                  f"action_schema={ACTION_SCHEMA}")
    else:
        print(f"Convert NPZ: workers={worker_count}, pending={len(pending)}")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_convert_job, pair, destination, config): (index, pair)
                for index, pair, destination in pending
            }
            for future in as_completed(futures):
                index, pair = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    raise RuntimeError(f"转换 Replay 失败：{pair.replay_id}") from error
                results.append(result)
                print(f"[{index}/{len(pairs)}] {result.replay_id}: frames={result.frames} "
                      f"transitions={result.transitions} card_overlap_cleaned={result.card_overlap_cleaned_rows} "
                      f"action_schema={ACTION_SCHEMA}")
        results.sort(key=lambda item: item.replay_id)
    report = {"schema": CQL_REPLAY_SCHEMA,
              "converted": [result.__dict__ for result in results]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
