from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.data.replay_reader import discover_replay_pairs, filter_character_replays
from soku_ai.data.split import create_split_manifest, save_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="按完整 Replay 创建固定 train/validation/test 划分")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    data = config["data"]
    destination = data["split_manifest"]
    from pathlib import Path

    if Path(destination).exists() and not arguments.force:
        raise FileExistsError(f"划分清单已经存在，使用 --force 才能重建: {destination}")
    pairs = filter_character_replays(
        discover_replay_pairs(data["raw_dir"]), int(data["suika_character_id"])
    )
    manifest = create_split_manifest(
        pairs,
        seed=int(config["training"]["seed"]),
        train_ratio=float(data["split_train_ratio"]),
        validation_ratio=float(data["split_validation_ratio"]),
        test_ratio=float(data["split_test_ratio"]),
    )
    save_split_manifest(manifest, destination)
    print(f"已写入 {destination}")
    for name, replay_ids in manifest["splits"].items():
        print(f"{name}: {len(replay_ids)} Replay")


if __name__ == "__main__":
    main()
