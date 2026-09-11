from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

from .schemas import ACTION_MAPPING_VERSION


class HorizontalAction(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = 2


class VerticalAction(IntEnum):
    NONE = 0
    UP = 1
    DOWN = 2


@dataclass(frozen=True)
class DecodedAction:
    horizontal: HorizontalAction
    vertical: VerticalAction
    a: bool
    b: bool
    c: bool
    d: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["horizontal"] = self.horizontal.name
        result["vertical"] = self.vertical.name
        result["A"] = result.pop("a")
        result["B"] = result.pop("b")
        result["C"] = result.pop("c")
        result["D"] = result.pop("d")
        return result


NUM_ACTIONS = 3 * 3 * 2 * 2 * 2 * 2


def encode_action(
    horizontal: HorizontalAction | int,
    vertical: VerticalAction | int,
    a: bool | int,
    b: bool | int,
    c: bool | int,
    d: bool | int,
) -> int:
    """使用固定进位顺序生成稳定的 0..143 动作编号。"""
    horizontal_value = int(HorizontalAction(horizontal))
    vertical_value = int(VerticalAction(vertical))
    action_id = horizontal_value
    action_id = action_id * 3 + vertical_value
    for pressed in (a, b, c, d):
        action_id = action_id * 2 + int(bool(pressed))
    return action_id


def decode_action(action_id: int) -> DecodedAction:
    if not 0 <= int(action_id) < NUM_ACTIONS:
        raise ValueError(f"动作编号必须位于 0..{NUM_ACTIONS - 1}: {action_id}")
    value = int(action_id)
    buttons: list[bool] = []
    for _ in range(4):
        buttons.append(bool(value % 2))
        value //= 2
    vertical = VerticalAction(value % 3)
    horizontal = HorizontalAction(value // 3)
    d, c, b, a = buttons
    return DecodedAction(horizontal, vertical, a, b, c, d)


def horizontal_from_raw(
    raw_value: int | float,
    character_direction: int | float,
    *,
    horizontal_positive_is_right: bool,
    direction_positive_faces_right: bool,
) -> HorizontalAction:
    if raw_value == 0:
        return HorizontalAction.NONE
    physical_right = raw_value > 0 if horizontal_positive_is_right else raw_value < 0
    facing_right = (
        character_direction > 0
        if direction_positive_faces_right
        else character_direction < 0
    )
    return HorizontalAction.FORWARD if physical_right == facing_right else HorizontalAction.BACKWARD


def vertical_from_raw(
    raw_value: int | float,
    *,
    vertical_positive_is_down: bool,
) -> VerticalAction:
    if raw_value == 0:
        return VerticalAction.NONE
    physical_down = raw_value > 0 if vertical_positive_is_down else raw_value < 0
    return VerticalAction.DOWN if physical_down else VerticalAction.UP


def encode_raw_input(
    horizontal_raw: int | float,
    vertical_raw: int | float,
    a_raw: int | float,
    b_raw: int | float,
    c_raw: int | float,
    d_raw: int | float,
    character_direction: int | float,
    *,
    horizontal_positive_is_right: bool,
    direction_positive_faces_right: bool,
    vertical_positive_is_down: bool,
) -> int:
    horizontal = horizontal_from_raw(
        horizontal_raw,
        character_direction,
        horizontal_positive_is_right=horizontal_positive_is_right,
        direction_positive_faces_right=direction_positive_faces_right,
    )
    vertical = vertical_from_raw(
        vertical_raw,
        vertical_positive_is_down=vertical_positive_is_down,
    )
    return encode_action(horizontal, vertical, a_raw != 0, b_raw != 0, c_raw != 0, d_raw != 0)


def action_mapping_metadata() -> dict[str, Any]:
    return {
        "version": ACTION_MAPPING_VERSION,
        "num_actions": NUM_ACTIONS,
        "radices": [3, 3, 2, 2, 2, 2],
        "field_order": ["horizontal", "vertical", "A", "B", "C", "D"],
        "horizontal": [member.name for member in HorizontalAction],
        "vertical": [member.name for member in VerticalAction],
    }

