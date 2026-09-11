from __future__ import annotations

import numpy as np
import pandas as pd

from soku_ai.data.preprocess import _build_actions, _build_episode_ids, _determine_sides


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row": [0, 1, 2, 4, 5],
            "battle_frame": [1, 2, 3, 1, 2],
            "current_round": [0, 0, 0, 1, 1],
            "left_character_id": [9] * 5,
            "right_character_id": [0] * 5,
            "left_input_horizontal": [0, 1, -1, 0, 1],
            "left_input_vertical": [0] * 5,
            "left_input_a": [0, 1, 0, 0, 0],
            "left_input_b": [0] * 5,
            "left_input_c": [0] * 5,
            "left_input_d": [0] * 5,
            "left_direction": [1] * 5,
        }
    )


def test_suika_side_conversion() -> None:
    assert _determine_sides(_frame(), 9) == ("left", "right")
    swapped = _frame().rename(
        columns={"left_character_id": "right_character_id", "right_character_id": "left_character_id"}
    )
    assert _determine_sides(swapped, 9) == ("right", "left")


def test_episode_boundary_and_action_shift() -> None:
    frame = _frame()
    _, starts, ends, _ = _build_episode_ids(frame)
    base = {
        "horizontal_positive_is_right": True,
        "direction_positive_faces_right": True,
        "vertical_positive_is_down": True,
    }
    actions0, valid0 = _build_actions(frame, "left", starts, ends, {**base, "action_shift": 0})
    actions1, valid1 = _build_actions(frame, "left", starts, ends, {**base, "action_shift": 1})
    assert np.array_equal(valid0, [True, True, False, True, False])
    assert np.array_equal(valid1, valid0)
    assert actions0[0] != actions1[0]

