from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


REWARD_ENGINE_VERSION = "modular_reward_v2_1"


@dataclass(frozen=True)
class RewardResult:
    rewards: np.ndarray
    dones: np.ndarray
    round_outcomes: np.ndarray
    training_weights: np.ndarray
    components: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, dict[str, float]]]
    wins: int
    losses: int

    def component_sums(self, valid_mask: np.ndarray) -> dict[str, float]:
        return {
            name: float(values[valid_mask].sum(dtype=np.float64))
            for name, values in self.components.items()
        }


def reward_config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {
            "engine_version": REWARD_ENGINE_VERSION,
            "config": config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Reward 配置 {name} 必须是映射")
    return value


def _float(config: Mapping[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"Reward 参数 {name} 必须是有限数")
    return value


def _player(frame: pd.DataFrame, side: str, field: str) -> np.ndarray:
    return frame[f"{side}_{field}"].to_numpy(dtype=np.float32, copy=False)


def _round_outcome(
    frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    self_side: str,
    opponent_side: str,
    next_index: int,
) -> int:
    source_row = int(frame.iloc[next_index]["source_row"])
    outcome_row = source_row + 1
    if outcome_row >= len(full_frame):
        return 0
    outcome = full_frame.iloc[outcome_row]
    self_score_gain = int(outcome[f"{self_side}_score"]) - int(
        frame.iloc[next_index][f"{self_side}_score"]
    )
    opponent_score_gain = int(outcome[f"{opponent_side}_score"]) - int(
        frame.iloc[next_index][f"{opponent_side}_score"]
    )
    if self_score_gain > opponent_score_gain:
        return 1
    if opponent_score_gain > self_score_gain:
        return -1
    self_hp = float(outcome[f"{self_side}_hp"])
    opponent_hp = float(outcome[f"{opponent_side}_hp"])
    if opponent_hp <= 0 < self_hp:
        return 1
    if self_hp <= 0 < opponent_hp:
        return -1
    return 0


def _legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    scale = _float(config, "damage_scale", 0.001)
    return {
        "version": str(config.get("version", "legacy_v1")),
        "damage": {
            "dealt_scale": scale,
            "taken_scale": scale,
        },
        "outcome": {
            "win_reward": _float(config, "round_win_reward", 10.0),
            "loss_penalty": abs(_float(config, "round_loss_reward", -10.0)),
        },
    }


def build_reward_result(
    *,
    frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    self_side: str,
    opponent_side: str,
    actions: np.ndarray,
    transition_valid: np.ndarray,
    episode_ids: np.ndarray,
    episode_start: np.ndarray,
    episode_end: np.ndarray,
    config: Mapping[str, Any],
) -> RewardResult:
    effective = config if "damage" in config else _legacy_config(config)
    damage = _section(effective, "damage")
    outcome = _section(effective, "outcome")
    positioning = _section(effective, "positioning")
    weighting = _section(effective, "demonstration_weighting")

    count = len(frame)
    components = {
        name: np.zeros(count, dtype=np.float32)
        for name in (
            "damage_dealt",
            "damage_taken",
            "self_corner_damage",
            "opponent_corner_damage",
            "hit_confirm",
            "approach",
            "far_retreat",
            "corner_escape",
            "corner_hold",
            "opponent_corner_pressure",
            "round_outcome",
        )
    }
    dones = np.zeros(count, dtype=np.bool_)
    round_outcomes = np.zeros(count, dtype=np.int8)
    valid_indices = np.flatnonzero(transition_valid)
    next_indices = valid_indices + 1

    self_hp = _player(frame, self_side, "hp")
    opponent_hp = _player(frame, opponent_side, "hp")
    dealt = np.zeros(count, dtype=np.float32)
    taken = np.zeros(count, dtype=np.float32)
    dealt[valid_indices] = np.maximum(
        0.0, opponent_hp[valid_indices] - opponent_hp[next_indices]
    )
    taken[valid_indices] = np.maximum(
        0.0, self_hp[valid_indices] - self_hp[next_indices]
    )

    terminal_indices = valid_indices[
        next_indices == episode_end[valid_indices].astype(np.int64) - 1
    ]
    wins = 0
    losses = 0
    for index in terminal_indices:
        next_index = int(index) + 1
        source_row = int(frame.iloc[next_index]["source_row"])
        outcome_row = source_row + 1
        if outcome_row < len(full_frame):
            final = full_frame.iloc[outcome_row]
            dealt[index] += max(
                0.0,
                float(opponent_hp[next_index])
                - float(final[f"{opponent_side}_hp"]),
            )
            taken[index] += max(
                0.0,
                float(self_hp[next_index]) - float(final[f"{self_side}_hp"]),
            )
        result = _round_outcome(
            frame,
            full_frame,
            self_side,
            opponent_side,
            next_index,
        )
        start = int(episode_start[index])
        end = int(episode_end[index])
        round_outcomes[start:end] = result
        dones[index] = True
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1

    self_x = _player(frame, self_side, "position_x")
    opponent_x = _player(frame, opponent_side, "position_x")
    self_y = _player(frame, self_side, "position_y")
    opponent_y = _player(frame, opponent_side, "position_y")
    distance = np.hypot(opponent_x - self_x, opponent_y - self_y)
    stage_min_x = _float(positioning, "stage_min_x", 40.0)
    stage_max_x = _float(positioning, "stage_max_x", 1240.0)
    if stage_max_x <= stage_min_x:
        raise ValueError("positioning.stage_max_x 必须大于 stage_min_x")
    corner_margin = max(0.0, _float(positioning, "corner_margin", 120.0))
    self_edge_distance = np.minimum(self_x - stage_min_x, stage_max_x - self_x)
    opponent_edge_distance = np.minimum(
        opponent_x - stage_min_x,
        stage_max_x - opponent_x,
    )
    self_corner = self_edge_distance <= corner_margin
    opponent_corner = opponent_edge_distance <= corner_margin

    dealt_scale = max(0.0, _float(damage, "dealt_scale", 0.001))
    taken_scale = max(0.0, _float(damage, "taken_scale", 0.001))
    components["damage_dealt"][valid_indices] = dealt[valid_indices] * dealt_scale
    components["damage_taken"][valid_indices] = -taken[valid_indices] * taken_scale
    self_corner_multiplier = max(
        1.0, _float(damage, "self_corner_taken_multiplier", 1.0)
    )
    opponent_corner_multiplier = max(
        1.0, _float(damage, "opponent_corner_dealt_multiplier", 1.0)
    )
    components["self_corner_damage"][valid_indices] = (
        -taken[valid_indices]
        * taken_scale
        * (self_corner_multiplier - 1.0)
        * self_corner[valid_indices]
    )
    components["opponent_corner_damage"][valid_indices] = (
        dealt[valid_indices]
        * dealt_scale
        * (opponent_corner_multiplier - 1.0)
        * opponent_corner[valid_indices]
    )
    hit_confirm_bonus = max(0.0, _float(damage, "hit_confirm_bonus", 0.0))
    components["hit_confirm"][valid_indices] = (
        dealt[valid_indices] > 0
    ).astype(np.float32) * hit_confirm_bonus

    action_values = np.asarray(actions, dtype=np.int64)
    horizontal = action_values // 48
    abc_pressed = np.bitwise_and(action_values % 16, 0b1110) != 0
    self_hitstop = _player(frame, self_side, "hitstop")
    can_shape = (
        transition_valid
        & (self_hitstop <= 0)
        & (taken <= 0)
    )
    distance_change = np.zeros(count, dtype=np.float32)
    distance_change[valid_indices] = (
        distance[next_indices] - distance[valid_indices]
    )
    edge_change = np.zeros(count, dtype=np.float32)
    edge_change[valid_indices] = (
        self_edge_distance[next_indices] - self_edge_distance[valid_indices]
    )

    approach_min_distance = max(
        0.0, _float(positioning, "approach_min_distance", 300.0)
    )
    position_delta_clip = max(
        0.0, _float(positioning, "position_delta_clip", 12.0)
    )
    approach_progress = np.clip(-distance_change, 0.0, position_delta_clip)
    approach_mask = (
        can_shape
        & (horizontal == 1)
        & (distance >= approach_min_distance)
    )
    components["approach"] = (
        approach_progress
        * approach_mask
        * max(0.0, _float(positioning, "approach_reward_per_unit", 0.0))
    ).astype(np.float32)

    retreat_min_distance = max(
        0.0, _float(positioning, "retreat_min_distance", 350.0)
    )
    retreat_progress = np.clip(distance_change, 0.0, position_delta_clip)
    retreat_mask = (
        can_shape
        & (horizontal == 2)
        & (distance >= retreat_min_distance)
    )
    components["far_retreat"] = (
        -retreat_progress
        * retreat_mask
        * max(0.0, _float(positioning, "retreat_penalty_per_unit", 0.0))
    ).astype(np.float32)

    escape_progress = np.clip(edge_change, 0.0, position_delta_clip)
    escape_mask = can_shape & self_corner & (horizontal == 1)
    components["corner_escape"] = (
        escape_progress
        * escape_mask
        * max(0.0, _float(positioning, "corner_escape_reward_per_unit", 0.0))
    ).astype(np.float32)

    corner_hold_penalty = max(
        0.0, _float(positioning, "corner_hold_penalty_per_frame", 0.0)
    )
    corner_hold_grace = max(
        0, int(positioning.get("corner_hold_grace_frames", 45))
    )
    # 墙角停滞同时覆盖无动作与继续后退；真实向场内移动时立即清空连续计数。
    corner_stagnant = (
        can_shape
        & self_corner
        & (edge_change <= 0.0)
        & ((horizontal == 0) | (horizontal == 2))
    )
    streak = 0
    last_episode = -1
    for index in valid_indices:
        episode = int(episode_ids[index])
        if episode != last_episode:
            streak = 0
            last_episode = episode
        streak = streak + 1 if corner_stagnant[index] else 0
        if streak > corner_hold_grace:
            components["corner_hold"][index] = -corner_hold_penalty

    pressure_mask = (
        can_shape
        & opponent_corner
        & ~self_corner
        & (distance <= max(0.0, _float(positioning, "pressure_max_distance", 350.0)))
        & ((horizontal == 1) | abc_pressed)
    )
    components["opponent_corner_pressure"][pressure_mask] = max(
        0.0, _float(positioning, "opponent_corner_pressure_reward", 0.0)
    )

    win_reward = _float(outcome, "win_reward", 10.0)
    loss_penalty = abs(_float(outcome, "loss_penalty", 10.0))
    for index in terminal_indices:
        if round_outcomes[index] > 0:
            components["round_outcome"][index] = win_reward
        elif round_outcomes[index] < 0:
            components["round_outcome"][index] = -loss_penalty

    rewards = np.zeros(count, dtype=np.float32)
    for values in components.values():
        rewards += values
    rewards[~transition_valid] = 0.0

    training_weights = np.ones(count, dtype=np.float32)
    training_weights[round_outcomes > 0] *= max(
        0.0, _float(weighting, "win_round_weight", 1.0)
    )
    training_weights[round_outcomes < 0] *= max(
        0.0, _float(weighting, "loss_round_weight", 1.0)
    )
    training_weights[(round_outcomes > 0) & abc_pressed] *= max(
        0.0, _float(weighting, "winning_attack_multiplier", 1.0)
    )
    training_weights[
        (round_outcomes < 0) & self_corner & (horizontal == 2)
    ] *= max(
        0.0, _float(weighting, "losing_corner_backward_multiplier", 1.0)
    )
    minimum_weight = max(1e-6, _float(weighting, "minimum_weight", 0.25))
    maximum_weight = max(
        minimum_weight, _float(weighting, "maximum_weight", 3.0)
    )
    training_weights = np.clip(
        training_weights,
        minimum_weight,
        maximum_weight,
    ).astype(np.float32)
    training_weights[~transition_valid] = 1.0

    def summarize(mask: np.ndarray) -> dict[str, float]:
        selected = transition_valid & mask
        return {
            "transitions": float(selected.sum()),
            "damage_dealt": float(dealt[selected].sum(dtype=np.float64)),
            "damage_taken": float(taken[selected].sum(dtype=np.float64)),
            "dealt_damage_frames": float((dealt[selected] > 0).sum()),
            "taken_damage_frames": float((taken[selected] > 0).sum()),
        }

    diagnostics = {
        "position_zones": {
            "deep_corner": summarize(self_edge_distance <= corner_margin * 0.5),
            "corner": summarize(
                (self_edge_distance > corner_margin * 0.5)
                & (self_edge_distance <= corner_margin)
            ),
            "side": summarize(
                (self_edge_distance > corner_margin)
                & (self_edge_distance <= corner_margin * 2.5)
            ),
            "center": summarize(self_edge_distance > corner_margin * 2.5),
        },
        "distance_bands": {
            "near": summarize(distance < 150.0),
            "middle": summarize((distance >= 150.0) & (distance < 300.0)),
            "far": summarize((distance >= 300.0) & (distance < 500.0)),
            "very_far": summarize(distance >= 500.0),
        },
    }

    return RewardResult(
        rewards=rewards,
        dones=dones,
        round_outcomes=round_outcomes,
        training_weights=training_weights,
        components=components,
        diagnostics=diagnostics,
        wins=wins,
        losses=losses,
    )
