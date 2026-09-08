from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
import yaml

from soku_cql.replay_preprocess import convert_dataset
from soku_cql.replay_collection import collect_replays


PROJECT_ROOT = _bootstrap.PROJECT


def _load_config(path: str) -> dict:
    candidate = Path(path)
    config_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 SokuDataBridge Replay CSV 转换为九宫格方向+六按钮的 CQL 分片"
    )
    parser.add_argument("--config", default="configs/replay_dataset.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-capture", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--conversion-workers", type=int)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if not arguments.skip_collection:
        collect_replays(
            _load_config(arguments.config),
            arguments.overwrite_capture,
            arguments.workers,
        )
    convert_dataset(
        arguments.config,
        arguments.overwrite,
        arguments.limit,
        arguments.conversion_workers,
    )


if __name__ == "__main__":
    main()
