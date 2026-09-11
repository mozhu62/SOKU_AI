from __future__ import annotations

from soku_ai.data.action_space import (
    NUM_ACTIONS,
    HorizontalAction,
    decode_action,
    encode_action,
    horizontal_from_raw,
)


def test_all_actions_round_trip() -> None:
    decoded = [decode_action(action_id) for action_id in range(NUM_ACTIONS)]
    rebuilt = {
        encode_action(item.horizontal, item.vertical, item.a, item.b, item.c, item.d)
        for item in decoded
    }
    assert rebuilt == set(range(NUM_ACTIONS))


def test_forward_backward_mirror() -> None:
    options = {
        "horizontal_positive_is_right": True,
        "direction_positive_faces_right": True,
    }
    assert horizontal_from_raw(1, 1, **options) == HorizontalAction.FORWARD
    assert horizontal_from_raw(-1, -1, **options) == HorizontalAction.FORWARD
    assert horizontal_from_raw(-1, 1, **options) == HorizontalAction.BACKWARD
    assert horizontal_from_raw(1, -1, **options) == HorizontalAction.BACKWARD

