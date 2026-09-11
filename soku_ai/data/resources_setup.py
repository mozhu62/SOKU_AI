from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .normalization import RunningMoments
from .resources_reader import read_resource_shard
from .resources_schema import OBSERVATION_VERSION, STATE_FIELDS, TACTICAL_FIELDS, HISTORY_FIELDS, OBJECT_FIELDS
from .split import manifest_hash, save_split_manifest, load_split_manifest

LOGGER = logging.getLogger(__name__)


@dataclass
class ResourceNormalization:
    payload: dict

    def normalize(self, values, group):
        mean = np.asarray(self.payload[f"{group}_mean"], np.float32)
        std = np.asarray(self.payload[f"{group}_std"], np.float32)
        return (values.astype(np.float32) - mean) / std

    def to_dict(self):
        return self.payload


def prepare_resources(config):
    """启动时核对固定划分并拟合必要统计；不写入原 NPZ，不生成第二份训练分片。"""
    if "_resources_prepared" in config:
        return config["_resources_prepared"]
    data = config["data"]
    root, source_split = Path(data["source_dir"]), Path(data["source_split_file"])
    if not root.is_dir() or not source_split.is_file():
        raise FileNotFoundError(f"需要已有 NPZ 目录和固定 BC/CQL 划分，不会自动重划分：{root}；{source_split}")
    source = json.loads(source_split.read_text(encoding="utf-8-sig"))
    canonical = {k: v for k, v in source.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if source.get("sha256") != digest or source.get("train_fraction") != 0.8:
        raise ValueError("当前固定划分的 SHA256 或 8:2 协议不兼容")
    files = source["files"]
    train, validation = source["train"], source["validation"]
    if (not train or not validation or len(train) != len(set(train)) or len(validation) != len(set(validation))
            or set(train) & set(validation) or set(train) | set(validation) != set(files)):
        raise ValueError("固定划分含空集、重叠、重复或遗漏")
    if {files[n] for n in train} & {files[n] for n in validation}:
        raise ValueError("固定划分的训练/验证存在相同文件内容")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*.npz")}
    if actual != set(files):
        raise ValueError("NPZ 文件集合与当前固定划分不一致；不会自动把新文件加入训练或重划分")
    manifest = {"version": 2, "source_split_sha256": source["sha256"], "source_files": files,
                "seed": source["seed"], "splits": {"train": train, "validation": validation, "test": []},
                "observation_schema": OBSERVATION_VERSION,
                "action_shift": data["action_shift"], "axes": {key: data[key] for key in
                 ("horizontal_positive_is_right", "vertical_positive_is_down", "direction_positive_faces_right")},
                "reward": dict(config["reward"])}
    manifest["sha256"] = manifest_hash(manifest)
    destination = Path(data["split_manifest"])
    norm_path = Path(data["normalization_file"])
    if (destination.resolve() == source_split.resolve() or norm_path.resolve() == source_split.resolve()
            or destination.resolve() == norm_path.resolve()
            or destination.resolve().is_relative_to(root.resolve()) or norm_path.resolve().is_relative_to(root.resolve())):
        raise ValueError("DQfD 派生清单和归一化必须保存在源数据目录之外")
    if destination.exists() and load_split_manifest(destination)["sha256"] != manifest["sha256"]:
        raise ValueError("DQfD 数据/奖励/方向定义已变化，请使用新的 split_manifest、normalization_file 和输出目录")

    moments = {"state": RunningMoments(27), "history": RunningMoments(11), "object": RunningMoments(8)}
    indices, train_names, ignored_cards = {}, set(train), 0
    for number, name in enumerate(sorted(files), 1):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"划分路径越界：{name}")
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != files[name]:
            raise ValueError(f"{name} 内容与固定划分不一致")
        shard = read_resource_shard(path, config)
        valid = shard["transition_valid"]
        indices[name] = np.flatnonzero(valid).astype(np.int32)
        ignored_cards += int(shard["ignored_card_frames"])
        if name in train_names:
            moments["state"].update(shard["state_continuous"][valid])
            moments["history"].update(shard["history_numerical"][shard["history_valid"]])
            for side in ("self", "opponent"):
                moments["object"].update(shard[f"{side}_object_numerical"])
        if number == 1 or number % 20 == 0 or number == len(files):
            LOGGER.info("核对当前 NPZ %d/%d：%s，有效转移 %d", number, len(files), name, len(indices[name]))
    payload = {"version": OBSERVATION_VERSION, "split_hash": manifest["sha256"],
               "state_features": list(STATE_FIELDS + TACTICAL_FIELDS), "history_features": list(HISTORY_FIELDS),
               "object_features": list(OBJECT_FIELDS), "fitted_replays": sorted(train)}
    for group, stats in moments.items():
        mean, std = stats.finalize()
        payload.update({f"{group}_mean": mean.tolist(), f"{group}_std": std.tolist()})
    for names in (train, validation):
        if not sum(len(indices[name]) for name in names):
            raise ValueError("训练或验证集合没有可用的连续转移")
    if norm_path.exists() and json.loads(norm_path.read_text(encoding="utf-8")) != payload:
        raise ValueError("归一化文件与当前训练数据不一致，请指定新的派生文件路径")
    if not destination.exists():
        save_split_manifest(manifest, destination)
    if not norm_path.exists():
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = norm_path.with_suffix(norm_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(norm_path)
    LOGGER.info("已沿用原固定划分：训练 %d / 验证 %d 个分片；卡牌按键投影到 ABCD 的帧数 %d",
                len(train), len(validation), ignored_cards)
    config["_resources_prepared"] = {"manifest": manifest, "indices": indices,
                                    "normalization": ResourceNormalization(payload)}
    return config["_resources_prepared"]
