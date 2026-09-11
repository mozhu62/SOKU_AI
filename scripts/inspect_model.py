from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401

from soku_ai.config import load_config
from soku_ai.models.factory import build_model
from soku_ai.inference.export import example_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="打印模型参数量、大小和主要模块输出 shape")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    model = build_model(config).eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    size_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1_048_576
    shapes: dict[str, object] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            shapes[name] = tuple(output.shape)
        return hook

    handles = [
        model.state_encoder.register_forward_hook(capture("state_encoder")),
        model.temporal_encoder.register_forward_hook(capture("temporal_encoder")),
        model.object_encoder.register_forward_hook(capture("object_encoder")),
        model.fusion.register_forward_hook(capture("fusion")),
        model.head.register_forward_hook(capture("dueling_head")),
    ]
    fields = ("state_continuous", "state_categorical", "history_numerical", "history_categorical", "history_mask",
              "self_object_numerical", "self_object_categorical", "self_object_mask",
              "opponent_object_numerical", "opponent_object_categorical", "opponent_object_mask")
    observation = dict(zip(fields, example_inputs(config, torch.device("cpu"))))
    with torch.no_grad():
        output = model(observation)
    for handle in handles:
        handle.remove()
    print(f"total_parameters={total:,}")
    print(f"trainable_parameters={trainable:,}")
    print(f"estimated_parameter_size_mb={size_mb:.2f}")
    for name, shape in shapes.items():
        print(f"{name}: {shape}")
    print(f"q_values: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
