from __future__ import annotations

import argparse
import json

import torch

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 checkpoint 元数据，不加载整个模型对象")
    parser.add_argument("checkpoint")
    arguments = parser.parse_args()
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    keys = (
        "checkpoint_version",
        "checkpoint_kind",
        "model_architecture_version",
        "observation_schema_version",
        "preprocessing_version",
        "training_step",
        "epoch",
        "random_seed",
        "dataset_split_manifest_hash",
        "best_metric",
        "provenance",
        "action_mapping",
    )
    print(json.dumps({key: checkpoint.get(key) for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
