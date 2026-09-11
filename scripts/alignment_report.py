from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.data.action_space import encode_raw_input
from soku_ai.data.replay_reader import discover_replay_pairs, read_main_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="输出输入变化附近的 state/action 对齐样本")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay-id")
    parser.add_argument("--changes", type=int, default=30)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    data = config["data"]
    pairs = discover_replay_pairs(data["raw_dir"])
    pair = next(
        (
            candidate
            for candidate in pairs
            if arguments.replay_id is None or candidate.replay_id == arguments.replay_id
        ),
        None,
    )
    if pair is None:
        raise ValueError("找不到指定 Replay")
    frame = read_main_replay(pair)
    suika_id = int(data["suika_character_id"])
    if (frame["left_character_id"] == suika_id).all():
        self_side, opponent_side = "left", "right"
    elif (frame["right_character_id"] == suika_id).all():
        self_side, opponent_side = "right", "left"
    else:
        raise ValueError("Replay 中 Suika 侧别不稳定")
    valid = frame[
        (frame["initialized"] != 0)
        & (frame["in_battle"] != 0)
        & frame["match_state"].isin(data["valid_match_states"])
        & frame["battle_sub_mode"].isin(data["valid_battle_sub_modes"])
    ].reset_index(drop=True)
    input_columns = [f"{self_side}_input_{name}" for name in ("horizontal", "vertical", "a", "b", "c", "d")]
    changes = valid[input_columns].ne(valid[input_columns].shift(1)).any(axis=1)
    change_indices = valid.index[changes].tolist()[: arguments.changes]
    selected = sorted(
        {
            index + offset
            for index in change_indices
            for offset in range(-arguments.context, arguments.context + 1)
            if 0 <= index + offset < len(valid)
        }
    )
    rows = []
    shift = int(data["action_shift"])
    for index in selected:
        action_index = min(index + shift, len(valid) - 1)
        action_row = valid.iloc[action_index]
        action_id = encode_raw_input(
            action_row[f"{self_side}_input_horizontal"],
            action_row[f"{self_side}_input_vertical"],
            action_row[f"{self_side}_input_a"],
            action_row[f"{self_side}_input_b"],
            action_row[f"{self_side}_input_c"],
            action_row[f"{self_side}_input_d"],
            action_row[f"{self_side}_direction"],
            horizontal_positive_is_right=data["horizontal_positive_is_right"],
            direction_positive_faces_right=data["direction_positive_faces_right"],
            vertical_positive_is_down=data["vertical_positive_is_down"],
        )
        row = valid.iloc[index]
        rows.append(
            {
                "row_index": index,
                "battle_frame": int(row["battle_frame"]),
                "current_round": int(row["current_round"]),
                "action_source_row": action_index,
                "action_shift": shift,
                "action_id": action_id,
                "self_position_x": row[f"{self_side}_position_x"],
                "self_position_y": row[f"{self_side}_position_y"],
                "opponent_position_x": row[f"{opponent_side}_position_x"],
                "opponent_position_y": row[f"{opponent_side}_position_y"],
                **{column.removeprefix(f"{self_side}_"): action_row[column] for column in input_columns},
            }
        )
    destination = Path(arguments.output or Path(data["reports_dir"]) / "alignment_report.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"Alignment report: {destination}")


if __name__ == "__main__":
    main()
