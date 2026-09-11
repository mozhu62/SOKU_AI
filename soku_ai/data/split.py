from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from .replay_reader import ReplayPair


def create_split_manifest(
    pairs: Sequence[ReplayPair],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"数据划分比例之和必须为 1，当前为 {total_ratio}")
    replay_ids = sorted(pair.replay_id for pair in pairs)
    random.Random(seed).shuffle(replay_ids)
    total = len(replay_ids)
    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)
    splits = {
        "train": sorted(replay_ids[:train_count]),
        "validation": sorted(replay_ids[train_count : train_count + validation_count]),
        "test": sorted(replay_ids[train_count + validation_count :]),
    }
    manifest = {
        "version": 1,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "splits": splits,
    }
    manifest["sha256"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    expected = manifest.get("sha256")
    actual = manifest_hash(manifest)
    if expected != actual:
        raise ValueError(f"数据划分清单校验失败: expected={expected}, actual={actual}")
    return manifest

