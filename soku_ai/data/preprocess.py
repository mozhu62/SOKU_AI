from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soku_ai.rewards import (
    REWARD_ENGINE_VERSION,
    build_reward_result,
    reward_config_hash,
)

from .action_space import NUM_ACTIONS, decode_action, encode_raw_input
from .object_reader import build_object_group_index
from .replay_reader import ReplayPair, read_main_replay, read_objects
from .sanitization import sanitize_object_numerical
from .schemas import (
    HISTORY_CATEGORICAL_FEATURES,
    HISTORY_NUMERICAL_FEATURES,
    OBJECT_CATEGORICAL_FEATURES,
    OBJECT_NUMERICAL_FEATURES,
    SCHEMA_VERSION,
    STATE_CATEGORICAL_FEATURES,
    STATE_CONTINUOUS_FEATURES,
)


@dataclass(frozen=True)
class PreprocessResult:
    replay_id: str
    shard_path: str
    frame_count: int
    transition_count: int
    episode_count: int
    suika_side: str
    opponent_character_id: int
    object_count: int
    truncated_object_count: int
    invalid_frame_count: int
    frame_gap_count: int
    action_histogram: tuple[int, ...]
    weather_histogram: dict[str, int]
    reward_sum: float
    reward_sum_squares: float
    reward_count: int
    reward_min: float
    reward_max: float
    average_objects_per_frame: float
    max_objects_per_frame: int
    average_round_length: float
    wins: int
    losses: int
    action_shift: int
    max_objects_per_side: int
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    reward_config_hash: str = ""
    reward_version: str = "legacy_v1"
    reward_engine_version: str = ""
    preprocessing_version: str = ""
    training_weight_sum: float = 0.0
    training_weight_sum_squares: float = 0.0
    training_weight_count: int = 0
    training_weight_min: float = 1.0
    training_weight_max: float = 1.0
    reward_diagnostics: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["action_histogram"] = list(self.action_histogram)
        return payload


def _player_values(frame: pd.DataFrame, side: str, field: str, dtype: Any = np.float32) -> np.ndarray:
    return frame[f"{side}_{field}"].to_numpy(dtype=dtype, copy=False)


def _determine_sides(frame: pd.DataFrame, suika_character_id: int) -> tuple[str, str]:
    left_is_suika = frame["left_character_id"].to_numpy(np.int64) == suika_character_id
    right_is_suika = frame["right_character_id"].to_numpy(np.int64) == suika_character_id
    if left_is_suika.all() and not right_is_suika.any():
        return "left", "right"
    if right_is_suika.all() and not left_is_suika.any():
        return "right", "left"
    if not left_is_suika.any() and not right_is_suika.any():
        raise ValueError("Replay 中不存在目标 Suika 角色")
    raise ValueError("Replay 内角色侧别不稳定，不能安全转换为第一视角")


def _build_episode_ids(valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    source_rows = valid["source_row"].to_numpy(np.int64)
    frames = valid["battle_frame"].to_numpy(np.int64)
    rounds = valid["current_round"].to_numpy(np.int64)
    new_episode = np.ones(len(valid), dtype=np.bool_)
    if len(valid) > 1:
        new_episode[1:] = (
            (source_rows[1:] != source_rows[:-1] + 1)
            | (frames[1:] != frames[:-1] + 1)
            | (rounds[1:] != rounds[:-1])
        )
    episode_ids = np.cumsum(new_episode, dtype=np.int32) - 1
    starts = np.flatnonzero(new_episode).astype(np.int32)
    ends = np.concatenate([starts[1:], np.asarray([len(valid)], dtype=np.int32)])
    start_for_frame = starts[episode_ids]
    end_for_frame = ends[episode_ids]
    contiguous_pairs = source_rows[1:] == source_rows[:-1] + 1
    frame_gap_count = int(
        (
            (frames[1:] != frames[:-1] + 1)
            & contiguous_pairs
            & (rounds[1:] == rounds[:-1])
        ).sum()
    )
    return episode_ids, start_for_frame, end_for_frame, frame_gap_count


def _build_state_arrays(
    frame: pd.DataFrame,
    self_side: str,
    opponent_side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    self_x = _player_values(frame, self_side, "position_x")
    self_y = _player_values(frame, self_side, "position_y")
    opponent_x = _player_values(frame, opponent_side, "position_x")
    opponent_y = _player_values(frame, opponent_side, "position_y")
    self_vx = _player_values(frame, self_side, "speed_x")
    self_vy = _player_values(frame, self_side, "speed_y")
    opponent_vx = _player_values(frame, opponent_side, "speed_x")
    opponent_vy = _player_values(frame, opponent_side, "speed_y")
    relative_x = opponent_x - self_x
    relative_y = opponent_y - self_y
    relative_vx = opponent_vx - self_vx
    relative_vy = opponent_vy - self_vy

    continuous_columns = {
        "self_position_x": self_x,
        "self_position_y": self_y,
        "self_speed_x": self_vx,
        "self_speed_y": self_vy,
        "self_direction": _player_values(frame, self_side, "direction"),
        "self_current_spirit": _player_values(frame, self_side, "current_spirit"),
        "self_action_frame_count": _player_values(frame, self_side, "action_frame_count"),
        "self_hitstop": _player_values(frame, self_side, "hitstop"),
        "self_combo_hits": _player_values(frame, self_side, "combo_hits"),
        "self_combo_limit": _player_values(frame, self_side, "combo_limit"),
        "self_total_object_count": _player_values(frame, self_side, "total_object_count"),
        "opponent_position_x": opponent_x,
        "opponent_position_y": opponent_y,
        "opponent_speed_x": opponent_vx,
        "opponent_speed_y": opponent_vy,
        "opponent_direction": _player_values(frame, opponent_side, "direction"),
        "opponent_current_spirit": _player_values(frame, opponent_side, "current_spirit"),
        "opponent_action_frame_count": _player_values(frame, opponent_side, "action_frame_count"),
        "opponent_hitstop": _player_values(frame, opponent_side, "hitstop"),
        "opponent_combo_hits": _player_values(frame, opponent_side, "combo_hits"),
        "opponent_combo_limit": _player_values(frame, opponent_side, "combo_limit"),
        "opponent_total_object_count": _player_values(frame, opponent_side, "total_object_count"),
        "weather_counter": frame["weather_counter"].to_numpy(np.float32),
        "relative_x": relative_x,
        "relative_y": relative_y,
        "relative_vx": relative_vx,
        "relative_vy": relative_vy,
        "distance_x": np.abs(relative_x),
        "distance_y": np.abs(relative_y),
        "euclidean_distance": np.sqrt(np.square(relative_x) + np.square(relative_y)),
    }
    categorical_columns = {
        "self_action": _player_values(frame, self_side, "action", np.int64),
        "self_action_block_id": _player_values(frame, self_side, "action_block_id", np.int64),
        "opponent_character_id": _player_values(frame, opponent_side, "character_id", np.int64),
        "opponent_action": _player_values(frame, opponent_side, "action", np.int64),
        "opponent_action_block_id": _player_values(frame, opponent_side, "action_block_id", np.int64),
        "active_weather": frame["active_weather"].to_numpy(np.int64),
        "displayed_weather": frame["displayed_weather"].to_numpy(np.int64),
        "stage_id": frame["stage_id"].to_numpy(np.int64),
    }
    state_continuous = np.column_stack(
        [continuous_columns[name] for name in STATE_CONTINUOUS_FEATURES]
    ).astype(np.float32, copy=False)
    state_categorical = np.column_stack(
        [categorical_columns[name] for name in STATE_CATEGORICAL_FEATURES]
    ).astype(np.int64, copy=False)

    history_columns = {
        "self_input_horizontal_raw": _player_values(frame, self_side, "input_horizontal"),
        "self_input_vertical_raw": _player_values(frame, self_side, "input_vertical"),
        "self_input_a_raw": _player_values(frame, self_side, "input_a"),
        "self_input_b_raw": _player_values(frame, self_side, "input_b"),
        "self_input_c_raw": _player_values(frame, self_side, "input_c"),
        "self_input_d_raw": _player_values(frame, self_side, "input_d"),
        "self_action_frame_count": continuous_columns["self_action_frame_count"],
        "opponent_action_frame_count": continuous_columns["opponent_action_frame_count"],
        "relative_x": relative_x,
        "relative_y": relative_y,
    }
    history_categorical_columns = {
        "self_action": categorical_columns["self_action"],
        "self_action_block_id": categorical_columns["self_action_block_id"],
        "opponent_action": categorical_columns["opponent_action"],
        "opponent_action_block_id": categorical_columns["opponent_action_block_id"],
    }
    history_numerical = np.column_stack(
        [history_columns[name] for name in HISTORY_NUMERICAL_FEATURES]
    ).astype(np.float32, copy=False)
    history_categorical = np.column_stack(
        [history_categorical_columns[name] for name in HISTORY_CATEGORICAL_FEATURES]
    ).astype(np.int64, copy=False)
    return state_continuous, state_categorical, history_numerical, history_categorical


def _build_actions(
    frame: pd.DataFrame,
    self_side: str,
    episode_start: np.ndarray,
    episode_end: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(frame)
    actions = np.zeros(count, dtype=np.int64)
    transition_valid = np.zeros(count, dtype=np.bool_)
    shift = int(config["action_shift"])
    if shift not in (0, 1):
        raise ValueError("action_shift 第一版只允许 0 或 1")
    horizontal = _player_values(frame, self_side, "input_horizontal")
    vertical = _player_values(frame, self_side, "input_vertical")
    button_a = _player_values(frame, self_side, "input_a")
    button_b = _player_values(frame, self_side, "input_b")
    button_c = _player_values(frame, self_side, "input_c")
    button_d = _player_values(frame, self_side, "input_d")
    direction = _player_values(frame, self_side, "direction")

    for index in range(max(0, count - 1)):
        if episode_start[index] != episode_start[index + 1]:
            continue
        action_index = index + shift
        if action_index >= episode_end[index]:
            continue
        transition_valid[index] = True
        actions[index] = encode_raw_input(
            horizontal[action_index],
            vertical[action_index],
            button_a[action_index],
            button_b[action_index],
            button_c[action_index],
            button_d[action_index],
            direction[action_index],
            horizontal_positive_is_right=bool(config["horizontal_positive_is_right"]),
            direction_positive_faces_right=bool(config["direction_positive_faces_right"]),
            vertical_positive_is_down=bool(config["vertical_positive_is_down"]),
        )
    return actions, transition_valid


def _pack_side_objects(
    frame: pd.DataFrame,
    objects: pd.DataFrame,
    group_indices: dict[tuple[int, int, str], np.ndarray],
    source_side: str,
    self_side: str,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    self_x = _player_values(frame, self_side, "position_x")
    self_y = _player_values(frame, self_side, "position_y")
    self_vx = _player_values(frame, self_side, "speed_x")
    self_vy = _player_values(frame, self_side, "speed_y")
    numerical_parts: list[np.ndarray] = []
    categorical_parts: list[np.ndarray] = []
    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    total_seen = 0
    truncated = 0

    serials = frame["sample_serial"].to_numpy(np.int64)
    battle_frames = frame["battle_frame"].to_numpy(np.int64)
    for frame_index, (serial, battle_frame) in enumerate(zip(serials, battle_frames, strict=True)):
        indices = group_indices.get((int(serial), int(battle_frame), source_side))
        if indices is None or len(indices) == 0:
            offsets[frame_index + 1] = offsets[frame_index]
            continue
        entity = objects.iloc[indices]
        relative_x = entity["position_x"].to_numpy(np.float32) - self_x[frame_index]
        relative_y = entity["position_y"].to_numpy(np.float32) - self_y[frame_index]
        distance = np.sqrt(np.square(relative_x) + np.square(relative_y))
        order = np.argsort(distance, kind="stable")
        total_seen += len(order)
        if len(order) > max_objects:
            truncated += len(order) - max_objects
            order = order[:max_objects]
        selected = entity.iloc[order]
        relative_x = relative_x[order]
        relative_y = relative_y[order]
        distance = distance[order]
        numeric_columns = {
            "relative_position_x": relative_x,
            "relative_position_y": relative_y,
            "relative_speed_x": selected["speed_x"].to_numpy(np.float32) - self_vx[frame_index],
            "relative_speed_y": selected["speed_y"].to_numpy(np.float32) - self_vy[frame_index],
            "gravity_x": selected["gravity_x"].to_numpy(np.float32),
            "gravity_y": selected["gravity_y"].to_numpy(np.float32),
            "direction": selected["direction"].to_numpy(np.float32),
            "action_frame_count": selected["action_frame_count"].to_numpy(np.float32),
            "hitstop": selected["hitstop"].to_numpy(np.float32),
            "hit_count": selected["hit_count"].to_numpy(np.float32),
            "hit_box_count": selected["hit_box_count"].to_numpy(np.float32),
            "hurt_box_count": selected["hurt_box_count"].to_numpy(np.float32),
            "frame_data_available": selected["frame_data_available"].to_numpy(np.float32),
            "frame_flags": selected["frame_flags"].to_numpy(np.float32),
            "attack_flags": selected["attack_flags"].to_numpy(np.float32),
            "frame_damage": selected["frame_damage"].to_numpy(np.float32),
            "frame_spirit_damage": selected["frame_spirit_damage"].to_numpy(np.float32),
            "distance_to_suika": distance,
        }
        categorical_columns = {
            "action": selected["action"].to_numpy(np.int64),
            "action_block_id": selected["action_block_id"].to_numpy(np.int64),
        }
        numerical = np.column_stack(
            [numeric_columns[name] for name in OBJECT_NUMERICAL_FEATURES]
        ).astype(np.float32, copy=False)
        numerical = sanitize_object_numerical(numerical)
        categorical = np.column_stack(
            [categorical_columns[name] for name in OBJECT_CATEGORICAL_FEATURES]
        ).astype(np.int64, copy=False)
        numerical_parts.append(numerical)
        categorical_parts.append(categorical)
        offsets[frame_index + 1] = offsets[frame_index] + len(selected)

    numerical_data = (
        np.concatenate(numerical_parts, axis=0)
        if numerical_parts
        else np.empty((0, len(OBJECT_NUMERICAL_FEATURES)), dtype=np.float32)
    )
    categorical_data = (
        np.concatenate(categorical_parts, axis=0)
        if categorical_parts
        else np.empty((0, len(OBJECT_CATEGORICAL_FEATURES)), dtype=np.int64)
    )
    return numerical_data, categorical_data, offsets, total_seen, truncated


def preprocess_replay(
    pair: ReplayPair,
    output_path: str | Path,
    data_config: dict[str, Any],
    reward_config: dict[str, Any],
) -> PreprocessResult:
    main = read_main_replay(
        pair,
        extra_columns=("left_score", "right_score"),
    )
    self_side, opponent_side = _determine_sides(main, int(data_config["suika_character_id"]))
    valid_mask = (
        (main["initialized"] != 0)
        & (main["in_battle"] != 0)
        & main["match_state"].isin(tuple(data_config["valid_match_states"]))
        & main["battle_sub_mode"].isin(tuple(data_config["valid_battle_sub_modes"]))
    )
    valid = main.loc[valid_mask].reset_index(drop=True)
    if len(valid) < 2:
        raise ValueError(f"有效战斗帧不足: {pair.replay_id}")

    episode_ids, episode_start, episode_end, frame_gap_count = _build_episode_ids(valid)
    state_continuous, state_categorical, history_numerical, history_categorical = _build_state_arrays(
        valid, self_side, opponent_side
    )
    actions, transition_valid = _build_actions(
        valid, self_side, episode_start, episode_end, data_config
    )
    reward_result = build_reward_result(
        frame=valid,
        full_frame=main,
        self_side=self_side,
        opponent_side=opponent_side,
        actions=actions,
        transition_valid=transition_valid,
        episode_ids=episode_ids,
        episode_start=episode_start,
        episode_end=episode_end,
        config=reward_config,
    )
    rewards = reward_result.rewards
    dones = reward_result.dones

    objects = read_objects(pair)
    group_indices = build_object_group_index(objects)
    max_objects = int(data_config["max_objects_per_side"])
    self_obj_num, self_obj_cat, self_offsets, self_seen, self_truncated = _pack_side_objects(
        valid, objects, group_indices, self_side, self_side, max_objects
    )
    opponent_obj_num, opponent_obj_cat, opponent_offsets, opponent_seen, opponent_truncated = _pack_side_objects(
        valid, objects, group_indices, opponent_side, self_side, max_objects
    )

    action_histogram = np.bincount(actions[transition_valid], minlength=NUM_ACTIONS)
    weather_values, weather_counts = np.unique(
        valid["displayed_weather"].to_numpy(np.int64), return_counts=True
    )
    weather_histogram = {
        str(int(value)): int(count)
        for value, count in zip(weather_values, weather_counts, strict=True)
    }
    valid_rewards = rewards[transition_valid]
    valid_training_weights = reward_result.training_weights[transition_valid]
    raw_objects_per_frame = (
        _player_values(valid, self_side, "total_object_count", np.int64)
        + _player_values(valid, opponent_side, "total_object_count", np.int64)
    )
    opponent_character_id = int(
        valid[f"{opponent_side}_character_id"].mode(dropna=False).iloc[0]
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": data_config["preprocessing_version"],
        "replay_id": pair.replay_id,
        "source_main": str(pair.main_path),
        "source_objects": str(pair.objects_path),
        "suika_side": self_side,
        "opponent_character_id": opponent_character_id,
        "action_shift": int(data_config["action_shift"]),
        "max_objects_per_side": max_objects,
        "reward_version": str(reward_config.get("version", "legacy_v1")),
        "reward_engine_version": REWARD_ENGINE_VERSION,
        "reward_config_hash": reward_config_hash(reward_config),
        "reward_components": sorted(reward_result.components),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reward_arrays = {
        f"reward_component_{name}": values
        for name, values in reward_result.components.items()
    }
    np.savez(
        destination,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        sample_serial=valid["sample_serial"].to_numpy(np.int64),
        battle_frame=valid["battle_frame"].to_numpy(np.int64),
        current_round=valid["current_round"].to_numpy(np.int32),
        episode_id=episode_ids.astype(np.int32),
        episode_start=episode_start.astype(np.int32),
        episode_end=episode_end.astype(np.int32),
        state_continuous=state_continuous,
        state_categorical=state_categorical,
        history_numerical=history_numerical,
        history_categorical=history_categorical,
        actions=actions,
        rewards=rewards,
        dones=dones,
        round_outcomes=reward_result.round_outcomes,
        training_weights=reward_result.training_weights,
        transition_valid=transition_valid,
        self_object_numerical=self_obj_num,
        self_object_categorical=self_obj_cat,
        self_object_offsets=self_offsets,
        opponent_object_numerical=opponent_obj_num,
        opponent_object_categorical=opponent_obj_cat,
        opponent_object_offsets=opponent_offsets,
        **reward_arrays,
    )
    return PreprocessResult(
        replay_id=pair.replay_id,
        shard_path=str(destination),
        frame_count=len(valid),
        transition_count=int(transition_valid.sum()),
        episode_count=int(episode_ids.max()) + 1,
        suika_side=self_side,
        opponent_character_id=opponent_character_id,
        object_count=self_seen + opponent_seen,
        truncated_object_count=self_truncated + opponent_truncated,
        invalid_frame_count=int((~valid_mask).sum()),
        frame_gap_count=frame_gap_count,
        action_histogram=tuple(int(value) for value in action_histogram),
        weather_histogram=weather_histogram,
        reward_sum=float(valid_rewards.sum()) if valid_rewards.size else 0.0,
        reward_sum_squares=float(np.square(valid_rewards.astype(np.float64)).sum()) if valid_rewards.size else 0.0,
        reward_count=int(valid_rewards.size),
        reward_min=float(valid_rewards.min()) if valid_rewards.size else 0.0,
        reward_max=float(valid_rewards.max()) if valid_rewards.size else 0.0,
        average_objects_per_frame=float(raw_objects_per_frame.mean()),
        max_objects_per_frame=int(raw_objects_per_frame.max(initial=0)),
        average_round_length=float(len(valid) / (int(episode_ids.max()) + 1)),
        wins=reward_result.wins,
        losses=reward_result.losses,
        action_shift=int(data_config["action_shift"]),
        max_objects_per_side=max_objects,
        reward_component_sums=reward_result.component_sums(transition_valid),
        reward_config_hash=reward_config_hash(reward_config),
        reward_version=str(reward_config.get("version", "legacy_v1")),
        reward_engine_version=REWARD_ENGINE_VERSION,
        preprocessing_version=str(data_config["preprocessing_version"]),
        training_weight_sum=float(valid_training_weights.sum(dtype=np.float64)),
        training_weight_sum_squares=float(
            np.square(valid_training_weights.astype(np.float64)).sum()
        ),
        training_weight_count=int(valid_training_weights.size),
        training_weight_min=(
            float(valid_training_weights.min())
            if valid_training_weights.size
            else 1.0
        ),
        training_weight_max=(
            float(valid_training_weights.max())
            if valid_training_weights.size
            else 1.0
        ),
        reward_diagnostics=reward_result.diagnostics,
    )


def _merge_reward_diagnostics(
    results: list[PreprocessResult],
) -> dict[str, dict[str, dict[str, float]]]:
    merged: dict[str, dict[str, dict[str, float]]] = {}
    for result in results:
        for group, buckets in result.reward_diagnostics.items():
            target_group = merged.setdefault(group, {})
            for bucket, values in buckets.items():
                target_bucket = target_group.setdefault(bucket, {})
                for name, value in values.items():
                    target_bucket[name] = target_bucket.get(name, 0.0) + float(value)
    for buckets in merged.values():
        for values in buckets.values():
            transitions = values.get("transitions", 0.0)
            values["damage_taken_per_1000_transitions"] = (
                values.get("damage_taken", 0.0) * 1000.0 / transitions
                if transitions
                else 0.0
            )
            values["taken_damage_frame_rate"] = (
                values.get("taken_damage_frames", 0.0) / transitions
                if transitions
                else 0.0
            )
    return merged


def write_dataset_report(results: list[PreprocessResult], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    actions = np.zeros(NUM_ACTIONS, dtype=np.int64)
    opponents: dict[str, int] = {}
    weather: dict[str, int] = {}
    reward_components: dict[str, float] = {}
    for result in results:
        actions += np.asarray(result.action_histogram, dtype=np.int64)
        key = str(result.opponent_character_id)
        opponents[key] = opponents.get(key, 0) + 1
        for weather_id, count in result.weather_histogram.items():
            weather[weather_id] = weather.get(weather_id, 0) + count
        for name, value in result.reward_component_sums.items():
            reward_components[name] = reward_components.get(name, 0.0) + float(value)

    total_objects = sum(result.object_count for result in results)
    truncated = sum(result.truncated_object_count for result in results)
    total_transitions = sum(result.transition_count for result in results)
    reward_count = sum(result.reward_count for result in results)
    reward_sum = sum(result.reward_sum for result in results)
    reward_sum_squares = sum(result.reward_sum_squares for result in results)
    reward_mean = reward_sum / reward_count if reward_count else 0.0
    reward_variance = max(reward_sum_squares / reward_count - reward_mean ** 2, 0.0) if reward_count else 0.0
    training_weight_count = sum(result.training_weight_count for result in results)
    training_weight_sum = sum(result.training_weight_sum for result in results)
    training_weight_sum_squares = sum(
        result.training_weight_sum_squares for result in results
    )
    training_weight_mean = (
        training_weight_sum / training_weight_count if training_weight_count else 0.0
    )
    training_weight_variance = (
        max(
            training_weight_sum_squares / training_weight_count
            - training_weight_mean ** 2,
            0.0,
        )
        if training_weight_count
        else 0.0
    )
    reward_diagnostics = _merge_reward_diagnostics(results)
    button_counts = {name: 0 for name in ("A", "B", "C", "D")}
    for action_id, count in enumerate(actions):
        decoded = decode_action(action_id)
        for name, pressed in zip(
            ("A", "B", "C", "D"),
            (decoded.a, decoded.b, decoded.c, decoded.d),
            strict=True,
        ):
            if pressed:
                button_counts[name] += int(count)

    payload = {
        "version": 1,
        "replay_count": len(results),
        "total_frames": sum(result.frame_count for result in results),
        "total_transitions": total_transitions,
        "suika_side": {
            "left": sum(result.suika_side == "left" for result in results),
            "right": sum(result.suika_side == "right" for result in results),
        },
        "opponent_character_distribution": opponents,
        "weather_distribution": weather,
        "action_distribution": actions.tolist(),
        "button_pressed_frequency": {
            key: (value / total_transitions if total_transitions else 0.0)
            for key, value in button_counts.items()
        },
        "total_objects_before_truncation": total_objects,
        "average_objects_per_frame": (
            sum(result.average_objects_per_frame * result.frame_count for result in results)
            / max(sum(result.frame_count for result in results), 1)
        ),
        "max_objects_per_frame": max((result.max_objects_per_frame for result in results), default=0),
        "truncated_objects": truncated,
        "object_truncation_rate": truncated / total_objects if total_objects else 0.0,
        "round_count": sum(result.episode_count for result in results),
        "average_round_length": (
            sum(result.frame_count for result in results)
            / max(sum(result.episode_count for result in results), 1)
        ),
        "wins": sum(result.wins for result in results),
        "losses": sum(result.losses for result in results),
        "reward_distribution": {
            "count": reward_count,
            "sum": reward_sum,
            "mean": reward_mean,
            "std": math.sqrt(reward_variance),
            "min": min((result.reward_min for result in results), default=0.0),
            "max": max((result.reward_max for result in results), default=0.0),
        },
        "reward_strategy": {
            "version": results[0].reward_version if results else None,
            "engine_version": results[0].reward_engine_version if results else None,
            "config_hash": results[0].reward_config_hash if results else None,
            "component_sums": reward_components,
        },
        "training_weight_distribution": {
            "count": training_weight_count,
            "mean": training_weight_mean,
            "std": math.sqrt(training_weight_variance),
            "min": min(
                (result.training_weight_min for result in results),
                default=1.0,
            ),
            "max": max(
                (result.training_weight_max for result in results),
                default=1.0,
            ),
        },
        "reward_diagnostics": reward_diagnostics,
        "invalid_frames": sum(result.invalid_frame_count for result in results),
        "frame_gaps": sum(result.frame_gap_count for result in results),
        "alignment_config": {
            "action_shift": results[0].action_shift if results else None,
            "max_objects_per_side": results[0].max_objects_per_side if results else None,
        },
        "replays": [result.to_dict() for result in results],
    }
    with (destination / "dataset_report.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    lines = [
        "# TH123 Suika 数据集报告",
        "",
        f"- Replay：{payload['replay_count']}",
        f"- 有效帧：{payload['total_frames']}",
        f"- Transition：{payload['total_transitions']}",
        f"- Round：{payload['round_count']}",
        f"- Suika 左/右：{payload['suika_side']['left']} / {payload['suika_side']['right']}",
        f"- 对象截断率：{payload['object_truncation_rate']:.6%}",
        f"- 平均/最大对象数：{payload['average_objects_per_frame']:.3f} / {payload['max_objects_per_frame']}",
        f"- 平均 Round 长度：{payload['average_round_length']:.1f} 帧",
        f"- Reward 均值/标准差：{payload['reward_distribution']['mean']:.6f} / {payload['reward_distribution']['std']:.6f}",
        f"- 训练权重均值/标准差：{payload['training_weight_distribution']['mean']:.6f} / {payload['training_weight_distribution']['std']:.6f}",
        f"- 训练权重范围：{payload['training_weight_distribution']['min']:.6f} / {payload['training_weight_distribution']['max']:.6f}",
        f"- 无效帧：{payload['invalid_frames']}",
        f"- 有效战斗帧缺口：{payload['frame_gaps']}",
        f"- 终局胜/负：{payload['wins']} / {payload['losses']}",
        "",
        "## Reward 分量累计值",
        "",
        *[
            f"- {name}：{value:.6f}"
            for name, value in sorted(reward_components.items())
        ],
        "",
        "## 位置风险统计",
        "",
        "| 区域 | Transition | 造成伤害 | 受到伤害 | 每千帧受伤 | 受伤帧率 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            "| {bucket} | {transitions:.0f} | {damage_dealt:.0f} | "
            "{damage_taken:.0f} | {damage_taken_per_1000_transitions:.2f} | "
            "{taken_damage_frame_rate:.3%} |".format(
                bucket=bucket,
                **values,
            )
            for bucket, values in reward_diagnostics.get("position_zones", {}).items()
        ],
        "",
        "完整 Action、天气、对手分布和逐 Replay 统计见 `dataset_report.json`。",
    ]
    (destination / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
