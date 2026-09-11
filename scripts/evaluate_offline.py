from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.data.resources_schema import enabled as resource_data_enabled
from soku_ai.models.factory import build_model
from soku_ai.training.checkpoint import load_checkpoint
from soku_ai.training.data_setup import build_processed_dataset
from soku_ai.training.evaluator import evaluate_offline
from soku_ai.training.utils import resolve_device, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 Demonstration 离线指标")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    set_random_seed(int(config["training"]["seed"]), deterministic=True)
    device = resolve_device(arguments.device)
    model = build_model(config).to(device)
    target = build_model(config).to(device)
    checkpoint = load_checkpoint(
        arguments.checkpoint,
        online_network=model,
        target_network=target,
        map_location=device,
    )
    dataset = build_processed_dataset(config, arguments.split)
    if not len(dataset):
        raise ValueError("指定的数据划分没有有效转移；resources_v4 沿用原 8:2 划分，请选择 validation")
    if resource_data_enabled(config):
        # 验证必须使用训练时相同的尺度，不能把另一批数据的归一化套到当前权重上。
        if checkpoint["normalization"] != dataset.normalization.to_dict():
            raise ValueError("验证数据、奖励定义或归一化与 checkpoint 不一致")
    metrics = evaluate_offline(
        model,
        dataset,
        device=device,
        batch_size=int(config["rl"]["batch_size"]),
        max_batches=max(1, (len(dataset) + int(config["rl"]["batch_size"]) - 1) // int(config["rl"]["batch_size"])),
        seed=int(config["training"]["seed"]),
        target_model=target,
        rl_config=config["rl"],
    )
    destination = Path(
        arguments.output
        or Path(config["data"]["reports_dir"]) / f"offline_{arguments.split}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
