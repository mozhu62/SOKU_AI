"""新卡牌原始 NPZ 转 BC 输入和用卡 Keyframe，不启动训练、不修改源档案。"""
import argparse
import json
from pathlib import Path
import sys

import _bootstrap
import numpy as np
import yaml

from soku_bc.card_capture import ARCHIVE_SCHEMA, training_fields
from soku_bc.spells import system_settings
from soku_bc.spell_data import prepare_spell_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="启用 spell_system 的 BC 配置，灵梦默认使用内置静态卡表")
    parser.add_argument("--cql-project", type=Path, default=_bootstrap.PROJECT.parent / "soku_cql")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spell = system_settings(cfg.get("spell_system"))
    if not spell["enabled"]:
        parser.error("请在 BC 配置中启用 spell_system")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("原始档案与训练输出目录必须相互独立")
    sys.path.insert(0, str(args.cql_project.resolve()))
    from soku_cql.binary_replay import records_to_tables, CARD_FRAME_DTYPE, validate_card_capture
    from soku_cql.replay_preprocess import convert_tables
    paths = sorted(source.rglob("*.npz"))
    if not paths:
        parser.error("没有找到原始 NPZ")
    card_ids = [card["id"] for card in spell["cards"]]
    conversion = {"data": {"suika_character_id": spell["character_id"],
        "nearest_projectiles_per_side": 3, "valid_match_states": [2], "valid_battle_sub_modes": [2],
        "action_shift": 1, "npz_compression_level": 1},
        # BC 不使用奖励；新数据占位为0，不涉及修改 IQL/CQL 的训练奖励定义。
        "reward": {"damage_dealt": 0.0, "damage_taken": 0.0}}
    for path in paths:
        destination = output / path.relative_to(source)
        if destination.exists():
            raise FileExistsError(f"禁止覆盖已有训练分片：{destination}")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            records = archive["records"]
        if (metadata.get("archive_schema") != ARCHIVE_SCHEMA or records.dtype != CARD_FRAME_DTYPE
                or metadata["capture"]["target_character_id"] != spell["character_id"]):
            raise ValueError(f"原始采集协议/角色不匹配：{path}")
        validate_card_capture(records["card_capture"], str(path))
        left = records["players"]["character_id"][:, 0] == spell["character_id"]
        right = records["players"]["character_id"][:, 1] == spell["character_id"]
        if not ((left.all() and not right.any()) or (right.all() and not left.any())):
            raise ValueError("目标角色视角不唯一或中途改变")
        fields, spell_meta = training_fields(records, 0 if left.all() else 1, card_ids, spell["character_id"])
        frame, objects, info = records_to_tables(records, metadata["capture"])
        temporary = Path(str(destination) + ".part")
        if temporary.exists():
            raise FileExistsError(f"存在上次未完成分片，请先检查：{temporary}")
        result = convert_tables(path.stem, frame, objects, temporary, conversion, info,
            extra_arrays=fields, extra_metadata={"spell_capture": spell_meta,
                "source_match_metadata": metadata["capture"].get("source_match_metadata")})
        with np.load(temporary, allow_pickle=False) as archive:
            shard = {key: archive[key] for key in archive.files if key != "metadata_json"}
            meta = json.loads(str(archive["metadata_json"].item()))
        prepare_spell_data(shard, meta, shard["transition_valid"], spell)
        temporary.rename(destination)
        supervised = shard["spell_supervision_mask"]
        print(f"{path.name}: frames={result.frames} spell_supervised={int(supervised.sum())} "
              f"use_keyframes={int((supervised & (shard['spell_target'] > 0)).sum())}")
    print("已生成 BC 分片。未创建 train/validation 划分、未拟合 normalization、未启动训练。")


if __name__ == "__main__":
    main()
