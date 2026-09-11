from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.training.demo_trainer import DemoTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="使用高手 Replay 进行 DQfD Demonstration 预训练")
    parser.add_argument("--config", default="configs/dqfd_suika_resources_v4.yaml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", help="恢复完整训练状态")
    checkpoint_group.add_argument(
        "--init-from",
        help="仅加载已有模型权重，重置优化器、Target 和 PER 后开始微调",
    )
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)
    if arguments.device:
        config["training"]["device"] = arguments.device
    if arguments.seed is not None:
        config["training"]["seed"] = arguments.seed
    DemoTrainer(
        config,
        resume_path=arguments.resume,
        init_from_path=arguments.init_from,
    ).train()


if __name__ == "__main__":
    main()
