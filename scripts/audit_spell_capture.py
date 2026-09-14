"""只读检查 NPZ 符卡采集能力；缺少确认事件时不伪造三类监督统计。"""
import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap
from soku_bc.spell_data import RAW_FIELDS, prepare_spell_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.npz"))
    if not files:
        raise ValueError("目录没有 NPZ")
    report = {"files": len(files), "confirmed_protocol_files": 0, "legacy_files": 0,
              "raw_hand_resource_files": 0, "raw_use_button_files": 0, "missing_examples": [],
              "passive_none": 0, "active_none": 0, "spell_use": 0, "cards": {},
              "examples": {"passive_none": [], "active_none": [], "spell_use": []}}
    for i, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as source:
            keys = set(source.files)
            report["raw_hand_resource_files"] += int({"self_hand_card_ids", "self_hand_card_costs", "self_card_state"} <= keys)
            report["raw_use_button_files"] += int("action_buttons" in keys)
            if not set(RAW_FIELDS) <= keys:
                report["legacy_files"] += 1
                if len(report["missing_examples"]) < 5:
                    report["missing_examples"].append({"file": path.name, "missing": sorted(set(RAW_FIELDS) - keys)})
                continue
            meta = json.loads(str(source["metadata_json"].item()))
            ids = meta["spell_capture"]["card_ids"]
            cfg = {"enabled": True, "character_id": meta["spell_capture"]["character_id"],
                   "cards": [{"id": x, "name": f"Card {x}"} for x in ids]}
            shard = {key: source[key] for key in (*RAW_FIELDS, "episode_id", "terminated", "transition_valid")}
        valid = shard["transition_valid"].astype(bool).copy()
        valid[-1] = False
        valid[:-1] &= shard["episode_id"][:-1] == shard["episode_id"][1:]
        valid[1:] &= ~shard["terminated"][:-1].astype(bool)
        prepare_spell_data(shard, meta, valid, cfg)
        available, target, supervised = (shard[key] for key in ("spell_available_mask", "spell_target", "spell_supervision_mask"))
        report["confirmed_protocol_files"] += 1
        masks = {"passive_none": valid & ~available.any(-1) & ~(supervised & (target > 0)),
                 "active_none": supervised & (target == 0), "spell_use": supervised & (target > 0)}
        for name, mask in masks.items():
            report[name] += int(mask.sum())
            for row in np.flatnonzero(mask)[:max(0, 5 - len(report["examples"][name]))]:
                report["examples"][name].append({"file": path.name, "row": int(row),
                    "available_card_ids": [ids[j] for j in np.flatnonzero(available[row])],
                    "target_card_id": ids[target[row]-1] if target[row] > 0 else None})
        for j, card in enumerate(ids):
            key = f"{cfg['character_id']}:{card}"
            count = report["cards"].setdefault(key, {"available": 0, "use": 0, "supervised_available": 0,
                                                   "use_when_available": 0})
            count["available"] += int((valid & available[:, j]).sum())
            count["supervised_available"] += int((supervised & available[:, j]).sum())
            count["use"] += int((supervised & (target == j + 1)).sum())
            count["use_when_available"] += int((supervised & available[:, j] & (target == j + 1)).sum())
        if i % 300 == 0:
            print(f"已核对 {i}/{len(files)}", flush=True)
    for count in report["cards"].values():
        denominator = count["supervised_available"]
        count["p_use_given_available"] = count["use_when_available"] / denominator if denominator else None
    report["complete_dataset_statistics"] = report["legacy_files"] == 0
    if not report["confirmed_protocol_files"]:
        for key in ("passive_none", "active_none", "spell_use", "cards"):
            report[key] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
