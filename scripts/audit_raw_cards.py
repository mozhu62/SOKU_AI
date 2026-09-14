"""只读统计游戏接口提供的可用集合、usedCards 新增记录与用卡 Keyframe。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import _bootstrap
import numpy as np
from soku_bc.card_capture import ARCHIVE_SCHEMA, use_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()
    totals = Counter()
    for path in sorted(args.directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            if meta.get("archive_schema") != ARCHIVE_SCHEMA:
                raise ValueError(f"不是游戏直接接口档案：{path}")
            records = archive["records"]
        cards = records["card_capture"]
        events = use_evidence(cards)
        totals.update(files=1, game_frames=len(records),
            unknown_event_player_frames=int((~events["event_valid"]).sum()),
            multiple_player_frames=int(events["multiple"].sum()),
            use_keyframes=int(events["keyframe"].sum()))
        for side in range(2):
            ids = events["used_card_id"][:, side][events["keyframe"][:, side]]
            print(f"{path.stem} {side + 1}P 使用卡ID次数：{dict(sorted(Counter(map(int, ids)).items()))}")
        for row, side in np.argwhere(events["keyframe"])[:max(0, args.examples)]:
            before = cards[row, side]["before"]
            print(json.dumps({"replay": path.stem, "row": int(row), "side": int(side + 1),
                "game_frame": int(records["battle_frame"][row]),
                "used_card_id": int(events["used_card_id"][row, side]),
                "available_before": before["card_ids"][:int(before["count"])].tolist()}, ensure_ascii=False))
    print(json.dumps(dict(totals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
