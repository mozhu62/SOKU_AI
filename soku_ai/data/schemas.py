from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final[str] = "suika_observation_v1"
ACTION_MAPPING_VERSION: Final[str] = "egocentric_3x3x2x2x2x2_v1"

STATE_CONTINUOUS_FEATURES: Final[tuple[str, ...]] = (
    "self_position_x",
    "self_position_y",
    "self_speed_x",
    "self_speed_y",
    "self_direction",
    "self_current_spirit",
    "self_action_frame_count",
    "self_hitstop",
    "self_combo_hits",
    "self_combo_limit",
    "self_total_object_count",
    "opponent_position_x",
    "opponent_position_y",
    "opponent_speed_x",
    "opponent_speed_y",
    "opponent_direction",
    "opponent_current_spirit",
    "opponent_action_frame_count",
    "opponent_hitstop",
    "opponent_combo_hits",
    "opponent_combo_limit",
    "opponent_total_object_count",
    "weather_counter",
    "relative_x",
    "relative_y",
    "relative_vx",
    "relative_vy",
    "distance_x",
    "distance_y",
    "euclidean_distance",
)

STATE_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "self_action",
    "self_action_block_id",
    "opponent_character_id",
    "opponent_action",
    "opponent_action_block_id",
    "active_weather",
    "displayed_weather",
    "stage_id",
)

HISTORY_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "self_input_horizontal_raw",
    "self_input_vertical_raw",
    "self_input_a_raw",
    "self_input_b_raw",
    "self_input_c_raw",
    "self_input_d_raw",
    "self_action_frame_count",
    "opponent_action_frame_count",
    "relative_x",
    "relative_y",
)

HISTORY_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "self_action",
    "self_action_block_id",
    "opponent_action",
    "opponent_action_block_id",
)

OBJECT_NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "relative_position_x",
    "relative_position_y",
    "relative_speed_x",
    "relative_speed_y",
    "gravity_x",
    "gravity_y",
    "direction",
    "action_frame_count",
    "hitstop",
    "hit_count",
    "hit_box_count",
    "hurt_box_count",
    "frame_data_available",
    "frame_flags",
    "attack_flags",
    "frame_damage",
    "frame_spirit_damage",
    "distance_to_suika",
)

OBJECT_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "action",
    "action_block_id",
)

PLAYER_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "character_id",
    "direction",
    "position_x",
    "position_y",
    "speed_x",
    "speed_y",
    "hp",
    "current_spirit",
    "action",
    "action_block_id",
    "action_frame_count",
    "hitstop",
    "combo_hits",
    "combo_limit",
    "total_object_count",
    "recorded_object_count",
    "input_horizontal",
    "input_vertical",
    "input_a",
    "input_b",
    "input_c",
    "input_d",
)

MAIN_GLOBAL_FIELDS: Final[tuple[str, ...]] = (
    "sample_serial",
    "battle_frame",
    "initialized",
    "in_battle",
    "scene_id",
    "battle_mode",
    "battle_sub_mode",
    "stage_id",
    "match_state",
    "current_round",
    "active_weather",
    "displayed_weather",
    "weather_counter",
)

OBJECT_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "sample_serial",
    "battle_frame",
    "side",
    "object_index",
    "total_object_count",
    "position_x",
    "position_y",
    "speed_x",
    "speed_y",
    "gravity_x",
    "gravity_y",
    "direction",
    "hp",
    "action",
    "action_block_id",
    "action_frame_count",
    "hitstop",
    "hit_count",
    "hurt_box_count",
    "hit_box_count",
    "frame_data_available",
    "frame_number",
    "frame_flags",
    "attack_flags",
    "frame_damage",
    "frame_spirit_damage",
)


def required_main_columns() -> tuple[str, ...]:
    players = tuple(
        f"{side}_{field}"
        for side in ("left", "right")
        for field in PLAYER_SOURCE_FIELDS
    )
    return MAIN_GLOBAL_FIELDS + players


@dataclass(frozen=True)
class ObservationShape:
    state_continuous: int = len(STATE_CONTINUOUS_FEATURES)
    state_categorical: int = len(STATE_CATEGORICAL_FEATURES)
    history_numerical: int = len(HISTORY_NUMERICAL_FEATURES)
    history_categorical: int = len(HISTORY_CATEGORICAL_FEATURES)
    object_numerical: int = len(OBJECT_NUMERICAL_FEATURES)
    object_categorical: int = len(OBJECT_CATEGORICAL_FEATURES)


OBSERVATION_SHAPE: Final[ObservationShape] = ObservationShape()

