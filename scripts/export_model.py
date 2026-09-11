from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from soku_ai.inference.export import export_torchscript


def main() -> None:
    parser = argparse.ArgumentParser(description="导出纯 state_dict 与 TorchScript 推理模型")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    state_dict_path = output_dir / "soku_q_network.state_dict.pt"
    torch.save(
        {
            "model_state_dict": checkpoint["model_state_dict"],
            "config": checkpoint["config"],
            "normalization": checkpoint["normalization"],
            "action_mapping": checkpoint["action_mapping"],
            "model_architecture_version": checkpoint["model_architecture_version"],
        },
        state_dict_path,
    )
    torchscript_path = export_torchscript(
        arguments.checkpoint,
        output_dir / "soku_q_network.torchscript.pt",
        device=arguments.device,
    )
    print(f"state_dict: {state_dict_path}")
    print(f"TorchScript: {torchscript_path}")


if __name__ == "__main__":
    main()

