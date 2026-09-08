from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap
import yaml
from filelock import FileLock

from soku_cql.config import load_config, resolve
from soku_cql.dataset import split_replays
from soku_cql.replay_preprocess import convert_dataset
from soku_cql.replay_collection import collect_replays


PROJECT_ROOT = _bootstrap.PROJECT


def _load_config(path: str) -> dict:
    candidate = Path(path)
    config_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 SokuDataBridge Replay CSV 转换为 CQL 分片，并准备固定训练/验证划分"
    )
    parser.add_argument("--config", default="configs/replay_dataset.yaml")
    parser.add_argument("--training-config", default="configs/cql_suika.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-capture", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--only-split", action="store_true", help="不采集或转换，只为现有 NPZ 生成/校验固定划分")
    parser.add_argument("--skip-split", action="store_true", help="转换完成后不生成固定划分")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--conversion-workers", type=int)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if arguments.only_split and arguments.skip_split:
        parser.error("--only-split 与 --skip-split 不能同时使用")
    if not arguments.only_split:
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
    if not arguments.skip_split:
        if arguments.limit is not None and not arguments.only_split:
            print("本次使用 --limit，仅转换部分数据，不生成固定划分。")
            return
        training_config = load_config(arguments.training_config)
        split_path = resolve(training_config["data"]["split_file"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        # 划分文件与训练入口使用同一把锁，避免转换结束和训练启动同时改写清单。
        with FileLock(str(split_path) + ".lock", timeout=0):
            split = split_replays(training_config, lambda message: print(f"[split] {message}"))
        print(f"固定划分已就绪：{split_path}；训练 {len(split['train'])}，验证 {len(split['validation'])}")


if __name__ == "__main__":
    main()
