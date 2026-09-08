from __future__ import annotations

from functools import lru_cache

import numpy as np


ACTION_SCHEMA = "soku_controller_joint432_v1"
ACTION_COUNT = 432
START_ACTION_ID = 432
PREVIOUS_ACTION_VOCAB = 433
NEUTRAL_ACTION_ID = 192
COMBAT_BUTTONS = ("melee", "dash", "light_projectile", "heavy_projectile")
CARD_COMMANDS = ("NONE", "CHANGE_CARD", "USE_CARD")


def _integer(value, name, low, high):
    array = np.asarray(value)
    if (not np.issubdtype(array.dtype, np.integer) or
            np.any((array < low) | (array > high))):
        raise ValueError(f"action schema 不兼容：{name} 必须是 {low}..{high} 的整数")
    return array.astype(np.int64, copy=False)


def encode(direction, combat_mask, card_command):
    """一个 ID 对应单游戏帧的完整控制器状态，方向始终是屏幕绝对九宫格。"""
    direction = _integer(direction, "direction", 1, 9)
    combat = _integer(combat_mask, "combat_mask", 0, 15)
    card = _integer(card_command, "card_command", 0, 2)
    result = (direction - 1) * 48 + combat * 3 + card
    return int(result) if result.ndim == 0 else result


def decode(joint_action_id):
    value = _integer(joint_action_id, "joint_action_id", 0, ACTION_COUNT - 1)
    result = (value // 48 + 1, (value % 48) // 3, value % 3)
    return tuple(int(x) for x in result) if value.ndim == 0 else result


def validate_buttons(buttons, context=""):
    values = np.asarray(buttons)
    if values.ndim < 1 or values.shape[-1] != 6 or not np.isin(values, [0, 1]).all():
        raise ValueError(f"action schema 不兼容：{context} 按钮必须是六列 0/1")
    conflict = (values[..., 4] == 1) & (values[..., 5] == 1)
    if np.any(conflict):
        positions = np.argwhere(conflict).tolist()[:8]
        raise ValueError(f"action schema 不兼容：{context} 同帧 CHANGE_CARD 与 USE_CARD 同时为 1，"
                         f"位置={positions}；三类 card_command 无法表示，禁止映射或丢弃")
    return values.astype(np.int64, copy=False)


def from_controller(direction, buttons, context=""):
    values = validate_buttons(buttons, context)
    # 固定 bit 顺序 A、D、B、C；切卡和用卡不是战斗 bit，而是互斥三分类。
    combat = (values[..., :4] * np.asarray([1, 2, 4, 8])).sum(-1)
    return encode(direction, combat, values[..., 4] + 2 * values[..., 5])


def from_raw_axes(horizontal, vertical, buttons, positive_down=True, context=""):
    h = _integer(horizontal, "action_horizontal", -1, 1)
    v = _integer(vertical, "action_vertical", -1, 1)
    direction = 5 + h - 3 * v * (1 if positive_down else -1)
    return from_controller(direction, buttons, context)


def to_controller(joint_action_id):
    direction, combat, card = decode(joint_action_id)
    combat, card = np.asarray(combat), np.asarray(card)
    buttons = np.stack([*((combat >> bit) & 1 for bit in range(4)), card == 1, card == 2], -1)
    return direction, buttons.astype(np.int64)


def action_name(joint_action_id):
    direction, combat, card = decode(joint_action_id)
    # 展示顺序 D+A+B+C 不改变存储 bit 顺序，6+D+A 表示同帧同时按下。
    names = [name for bit, name in ((1, "D"), (0, "A"), (2, "B"), (3, "C")) if combat & (1 << bit)]
    if card:
        names.append(CARD_COMMANDS[card])
    return "+".join([str(direction), *names])


@lru_cache(maxsize=1)
def _action_catalog():
    return tuple(action_name(i) for i in range(ACTION_COUNT))


def action_catalog():
    return list(_action_catalog())


def frequency_rows(counts, limit=20):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    order = np.argsort(-counts, kind="stable")[:limit]
    return [{"joint_action_id": int(i), "action": _action_catalog()[i], "count": int(counts[i]),
             "fraction": int(counts[i]) / total if total else None} for i in order if counts[i] > 0]


def previous_actions(joint, duration, episode, valid, terminated):
    """在完整分片上先右移一帧，再取训练序列；绝不将 action_t 回灌到 observation_t。"""
    joint = _integer(joint, "joint_action_id", 0, ACTION_COUNT - 1)
    duration = _integer(duration, "action_duration", 1, np.iinfo(np.int64).max)
    count = len(joint)
    if any(np.asarray(x).shape != (count,) for x in (duration, episode, valid, terminated)):
        raise ValueError("上一帧动作历史数组长度不一致")
    previous = np.full(count, START_ACTION_ID, np.int64)
    age = np.zeros((count, 1), np.float32)
    connected = ((episode[1:] == episode[:-1]) & valid[:-1] & ~terminated[:-1])
    positions = np.flatnonzero(connected) + 1
    previous[positions] = joint[positions - 1]
    # duration 仅描述水平/垂直组合的持续帧数，按钮变化不重置该计数。
    age[positions, 0] = np.minimum(duration[positions - 1], 60) / 60.0
    return previous, age


BUTTON_COLUMNS = (
    "input_a",
    "input_d",
    "input_b",
    "input_c",
    "input_change_card",
    "input_spell_card",
)


def compact_actions(
    frame,
    side: str,
    source_index: np.ndarray,
    episode: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """保留紧凑原始轴/方向持续帧数/按钮列；训练时统一编码为 joint_action_id。"""
    raw_horizontal = frame[f"{side}_input_horizontal"].to_numpy(np.int16)
    raw_vertical = frame[f"{side}_input_vertical"].to_numpy(np.int16)
    horizontal = np.sign(raw_horizontal).astype(np.int8)
    vertical = np.sign(raw_vertical).astype(np.int8)

    indices = np.arange(len(frame), dtype=np.int32)
    starts = np.ones(len(frame), dtype=np.bool_)
    if len(frame) > 1:
        starts[1:] = (
            (episode[1:] != episode[:-1])
            | (horizontal[1:] != horizontal[:-1])
            | (vertical[1:] != vertical[:-1])
        )
    run_starts = np.maximum.accumulate(np.where(starts, indices, 0))
    duration = indices - run_starts + 1

    buttons = np.column_stack([
        frame[f"{side}_{column}"].to_numpy(np.int16) > 0
        for column in BUTTON_COLUMNS
    ]).astype(np.uint8)
    validate_buttons(buttons, f"{side} 原始输入帧")

    return (
        horizontal[source_index],
        vertical[source_index],
        duration[source_index],
        buttons[source_index],
    )
