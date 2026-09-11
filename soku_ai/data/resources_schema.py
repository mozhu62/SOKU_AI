"""当前原始 NPZ 与 DQfD 输入之间的显式契约，不依赖 BC/CQL 包。"""
from .schemas import ObservationShape

DATASET_SCHEMA = "soku_cql_raw_axes_action_resources_v4"
OBSERVATION_VERSION = "dqfd_resources_v4_observation_v1"
MODEL_TYPE = "tcn_entity_dueling_dqn_resources_v4"
ARCHITECTURE_VERSION = "tcn_entity_dueling_dqn_resources_v4_v1"
STATE_FIELDS = (
    "self_position_x", "self_position_y", "self_speed_x", "self_speed_y",
    "self_direction", "self_current_spirit", "self_action_frame_count",
    "opponent_position_x", "opponent_position_y", "opponent_speed_x", "opponent_speed_y",
    "opponent_direction", "opponent_current_spirit", "opponent_action_frame_count",
    "relative_x", "relative_y", "self_hp", "opponent_hp",
)
CATEGORY_FIELDS = ("self_action", "self_action_block_id", "opponent_action", "opponent_action_block_id", "active_weather")
TACTICAL_FIELDS = ("self_guarding", "opponent_guarding", "self_graze_active", "opponent_graze_active",
                   "opponent_projectile_attack_active", "self_hurt_state", "self_airborne_flag",
                   "opponent_hurt_state", "opponent_airborne_flag")
HISTORY_FIELDS = ("previous_horizontal", "previous_vertical", "previous_a", "previous_b", "previous_c",
                  "previous_d", "previous_axes_duration", "self_action_frame_count", "opponent_action_frame_count",
                  "relative_x", "relative_y")
OBJECT_FIELDS = ("relative_position_x", "relative_position_y", "relative_speed_x", "relative_speed_y",
                 "direction", "action_frame_count", "hitstop", "hit_count")
BUTTON_FIELDS = ("melee", "dash", "light_projectile", "heavy_projectile", "change_card", "use_spell_card")
SHAPE = ObservationShape(state_continuous=27, state_categorical=5, history_numerical=11,
                         history_categorical=4, object_numerical=8, object_categorical=2)


def enabled(config):
    return config.get("data", {}).get("format") == "resources_v4"
