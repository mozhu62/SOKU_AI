from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from soku_ai.data.normalization import NormalizationStats, normalization_from_dict
from soku_ai.data.sanitization import sanitize_object_numerical
from soku_ai.data.schemas import (
    HISTORY_CATEGORICAL_FEATURES,
    HISTORY_NUMERICAL_FEATURES,
    OBJECT_CATEGORICAL_FEATURES,
    OBJECT_NUMERICAL_FEATURES,
    STATE_CATEGORICAL_FEATURES,
    STATE_CONTINUOUS_FEATURES,
)
from soku_ai.data.transition_builder import CATEGORICAL_PADDING_VALUE

from .shared_state import ObjectState, PlayerState, StatePayload


class LiveStateUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveObservation:
    observation: dict[str, torch.Tensor]
    game_process_id: int
    sample_serial: int
    battle_frame: int
    current_round: int
    self_side: str
    self_direction: int
    facing_right: bool
    self_hitstop: int
    history_count: int


class LiveObservationBuilder:
    """把共享内存快照转换成与 Replay Dataset 完全同构的模型输入。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        normalization_payload: Mapping[str, object],
    ) -> None:
        if config["data"].get("format") == "resources_v4":
            # 新旧输入有不同列序，运行器根据模型版本选择观察构造器。
            raise ValueError(
                "resources_v4 模型应使用 ResourceLiveObservationBuilder，不能直接调用旧观察构造器"
            )
        data_config = config["data"]
        self.suika_character_id = int(data_config["suika_character_id"])
        self.history_len = int(data_config["history_len"])
        self.max_objects = int(data_config["max_objects_per_side"])
        self.direction_positive_faces_right = bool(
            data_config["direction_positive_faces_right"]
        )
        self.valid_match_states = {
            int(value) for value in data_config.get("valid_match_states", (2,))
        }
        self.normalization: NormalizationStats = normalization_from_dict(
            normalization_payload
        )
        self._history_numerical: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._history_categorical: deque[np.ndarray] = deque(maxlen=self.history_len)
        self._last_frame_key: tuple[int, int, int] | None = None

    def reset(self) -> None:
        self._history_numerical.clear()
        self._history_categorical.clear()
        self._last_frame_key = None

    def _select_players(
        self,
        payload: StatePayload,
    ) -> tuple[PlayerState, PlayerState, str]:
        left_is_self = int(payload.left.characterId) == self.suika_character_id
        right_is_self = int(payload.right.characterId) == self.suika_character_id
        if left_is_self == right_is_self:
            raise LiveStateUnavailable(
                "战斗中必须恰好有一名萃香，当前无法确定模型控制侧"
            )
        if left_is_self:
            return payload.left, payload.right, "left"
        return payload.right, payload.left, "right"

    @staticmethod
    def _state_arrays(
        payload: StatePayload,
        self_player: PlayerState,
        opponent: PlayerState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        relative_x = float(opponent.positionX) - float(self_player.positionX)
        relative_y = float(opponent.positionY) - float(self_player.positionY)
        relative_vx = float(opponent.speedX) - float(self_player.speedX)
        relative_vy = float(opponent.speedY) - float(self_player.speedY)
        continuous_values = {
            "self_position_x": self_player.positionX,
            "self_position_y": self_player.positionY,
            "self_speed_x": self_player.speedX,
            "self_speed_y": self_player.speedY,
            "self_direction": self_player.direction,
            "self_current_spirit": self_player.currentSpirit,
            "self_action_frame_count": self_player.actionFrameCount,
            "self_hitstop": self_player.hitstop,
            "self_combo_hits": self_player.comboHits,
            "self_combo_limit": self_player.comboLimit,
            "self_total_object_count": self_player.totalObjectCount,
            "opponent_position_x": opponent.positionX,
            "opponent_position_y": opponent.positionY,
            "opponent_speed_x": opponent.speedX,
            "opponent_speed_y": opponent.speedY,
            "opponent_direction": opponent.direction,
            "opponent_current_spirit": opponent.currentSpirit,
            "opponent_action_frame_count": opponent.actionFrameCount,
            "opponent_hitstop": opponent.hitstop,
            "opponent_combo_hits": opponent.comboHits,
            "opponent_combo_limit": opponent.comboLimit,
            "opponent_total_object_count": opponent.totalObjectCount,
            "weather_counter": payload.weatherCounter,
            "relative_x": relative_x,
            "relative_y": relative_y,
            "relative_vx": relative_vx,
            "relative_vy": relative_vy,
            "distance_x": abs(relative_x),
            "distance_y": abs(relative_y),
            "euclidean_distance": math.hypot(relative_x, relative_y),
        }
        categorical_values = {
            "self_action": self_player.action,
            "self_action_block_id": self_player.actionBlockId,
            "opponent_character_id": opponent.characterId,
            "opponent_action": opponent.action,
            "opponent_action_block_id": opponent.actionBlockId,
            "active_weather": payload.activeWeather,
            "displayed_weather": payload.displayedWeather,
            "stage_id": payload.stageId,
        }
        history_values = {
            "self_input_horizontal_raw": self_player.inputHorizontal,
            "self_input_vertical_raw": self_player.inputVertical,
            "self_input_a_raw": self_player.inputA,
            "self_input_b_raw": self_player.inputB,
            "self_input_c_raw": self_player.inputC,
            "self_input_d_raw": self_player.inputD,
            "self_action_frame_count": self_player.actionFrameCount,
            "opponent_action_frame_count": opponent.actionFrameCount,
            "relative_x": relative_x,
            "relative_y": relative_y,
        }
        history_categorical_values = {
            "self_action": self_player.action,
            "self_action_block_id": self_player.actionBlockId,
            "opponent_action": opponent.action,
            "opponent_action_block_id": opponent.actionBlockId,
        }
        state_numerical = np.asarray(
            [continuous_values[name] for name in STATE_CONTINUOUS_FEATURES],
            dtype=np.float32,
        )
        state_categorical = np.asarray(
            [categorical_values[name] for name in STATE_CATEGORICAL_FEATURES],
            dtype=np.int64,
        )
        history_numerical = np.asarray(
            [history_values[name] for name in HISTORY_NUMERICAL_FEATURES],
            dtype=np.float32,
        )
        history_categorical = np.asarray(
            [history_categorical_values[name] for name in HISTORY_CATEGORICAL_FEATURES],
            dtype=np.int64,
        )
        if not np.isfinite(state_numerical).all() or not np.isfinite(
            history_numerical
        ).all():
            raise LiveStateUnavailable("实时角色状态中发现 NaN 或 Inf")
        return (
            state_numerical,
            state_categorical,
            history_numerical,
            history_categorical,
        )

    def _object_arrays(
        self,
        player: PlayerState,
        self_player: PlayerState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates: list[tuple[float, int, ObjectState]] = []
        count = min(int(player.objectCount), len(player.objects))
        for index in range(count):
            entity = player.objects[index]
            relative_x = float(entity.positionX) - float(self_player.positionX)
            relative_y = float(entity.positionY) - float(self_player.positionY)
            candidates.append((math.hypot(relative_x, relative_y), index, entity))
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = candidates[: self.max_objects]

        numerical = np.zeros(
            (self.max_objects, len(OBJECT_NUMERICAL_FEATURES)), dtype=np.float32
        )
        categorical = np.full(
            (self.max_objects, len(OBJECT_CATEGORICAL_FEATURES)),
            CATEGORICAL_PADDING_VALUE,
            dtype=np.int64,
        )
        mask = np.zeros(self.max_objects, dtype=np.bool_)
        if not selected:
            return numerical, categorical, mask

        rows: list[list[float]] = []
        categories: list[list[int]] = []
        for distance, _, entity in selected:
            relative_x = float(entity.positionX) - float(self_player.positionX)
            relative_y = float(entity.positionY) - float(self_player.positionY)
            values = {
                "relative_position_x": relative_x,
                "relative_position_y": relative_y,
                "relative_speed_x": float(entity.speedX) - float(self_player.speedX),
                "relative_speed_y": float(entity.speedY) - float(self_player.speedY),
                "gravity_x": entity.gravityX,
                "gravity_y": entity.gravityY,
                "direction": entity.direction,
                "action_frame_count": entity.actionFrameCount,
                "hitstop": entity.hitstop,
                "hit_count": entity.hitCount,
                "hit_box_count": entity.hitBoxCount,
                "hurt_box_count": entity.hurtBoxCount,
                "frame_data_available": entity.frameDataAvailable,
                "frame_flags": entity.frameFlags,
                "attack_flags": entity.attackFlags,
                "frame_damage": entity.frameDamage,
                "frame_spirit_damage": entity.frameSpiritDamage,
                "distance_to_suika": distance,
            }
            rows.append([float(values[name]) for name in OBJECT_NUMERICAL_FEATURES])
            categories.append([int(entity.action), int(entity.actionBlockId)])

        valid_numerical = sanitize_object_numerical(
            np.asarray(rows, dtype=np.float32)
        )
        valid_numerical = self.normalization.normalize_objects(valid_numerical)
        valid_count = len(selected)
        numerical[:valid_count] = valid_numerical
        categorical[:valid_count] = np.asarray(categories, dtype=np.int64)
        mask[:valid_count] = True
        return numerical, categorical, mask

    def _history_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        numerical = np.zeros(
            (self.history_len, len(HISTORY_NUMERICAL_FEATURES)), dtype=np.float32
        )
        categorical = np.full(
            (self.history_len, len(HISTORY_CATEGORICAL_FEATURES)),
            CATEGORICAL_PADDING_VALUE,
            dtype=np.int64,
        )
        mask = np.zeros(self.history_len, dtype=np.bool_)
        count = len(self._history_numerical)
        if count:
            destination = slice(self.history_len - count, self.history_len)
            raw_numerical = np.stack(tuple(self._history_numerical), axis=0)
            numerical[destination] = self.normalization.normalize_history(raw_numerical)
            categorical[destination] = np.stack(
                tuple(self._history_categorical), axis=0
            )
            mask[destination] = True
        return numerical, categorical, mask

    def build(self, payload: StatePayload) -> LiveObservation | None:
        if not int(payload.initialized) or not int(payload.inBattle):
            self.reset()
            raise LiveStateUnavailable("等待进入战斗")
        if int(payload.matchState) not in self.valid_match_states:
            self.reset()
            raise LiveStateUnavailable(
                f"等待实际战斗帧，当前 match_state={int(payload.matchState)}"
            )

        self_player, opponent, self_side = self._select_players(payload)
        frame_key = (
            int(payload.gameProcessId),
            int(payload.currentRound),
            int(payload.battleFrame),
        )
        if frame_key == self._last_frame_key:
            return None
        if self._last_frame_key is not None:
            last_pid, last_round, last_frame = self._last_frame_key
            same_round = frame_key[0] == last_pid and frame_key[1] == last_round
            frame_gap = frame_key[2] - last_frame
            if not same_round or frame_gap <= 0 or frame_gap > self.history_len:
                self._history_numerical.clear()
                self._history_categorical.clear()
            elif frame_gap > 1 and self._history_numerical:
                # 共享内存只保存最新帧；少量漏帧时沿用上一输入，避免历史长期只有一帧。
                repeated_numerical = self._history_numerical[-1]
                repeated_categorical = self._history_categorical[-1]
                for _ in range(frame_gap - 1):
                    self._history_numerical.append(repeated_numerical.copy())
                    self._history_categorical.append(repeated_categorical.copy())

        (
            state_numerical,
            state_categorical,
            history_numerical_row,
            history_categorical_row,
        ) = self._state_arrays(payload, self_player, opponent)
        self._history_numerical.append(history_numerical_row)
        self._history_categorical.append(history_categorical_row)
        history_numerical, history_categorical, history_mask = self._history_arrays()
        self_object_numerical, self_object_categorical, self_object_mask = (
            self._object_arrays(self_player, self_player)
        )
        (
            opponent_object_numerical,
            opponent_object_categorical,
            opponent_object_mask,
        ) = self._object_arrays(opponent, self_player)
        self._last_frame_key = frame_key

        normalized_state = self.normalization.normalize_state(state_numerical)
        observation = {
            "state_continuous": torch.from_numpy(normalized_state),
            "state_categorical": torch.from_numpy(state_categorical),
            "history_numerical": torch.from_numpy(history_numerical),
            "history_categorical": torch.from_numpy(history_categorical),
            "history_mask": torch.from_numpy(history_mask),
            "self_object_numerical": torch.from_numpy(self_object_numerical),
            "self_object_categorical": torch.from_numpy(self_object_categorical),
            "self_object_mask": torch.from_numpy(self_object_mask),
            "opponent_object_numerical": torch.from_numpy(
                opponent_object_numerical
            ),
            "opponent_object_categorical": torch.from_numpy(
                opponent_object_categorical
            ),
            "opponent_object_mask": torch.from_numpy(opponent_object_mask),
        }
        direction = int(self_player.direction)
        facing_right = (
            direction > 0
            if self.direction_positive_faces_right
            else direction < 0
        )
        return LiveObservation(
            observation=observation,
            game_process_id=int(payload.gameProcessId),
            sample_serial=int(payload.sampleSerial),
            battle_frame=int(payload.battleFrame),
            current_round=int(payload.currentRound),
            self_side=self_side,
            self_direction=direction,
            facing_right=facing_right,
            self_hitstop=int(self_player.hitstop),
            history_count=len(self._history_numerical),
        )
