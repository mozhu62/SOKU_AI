from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.data.resources_schema import MODEL_TYPE, SHAPE
from soku_ai.data.transition_builder import CATEGORICAL_PADDING_VALUE
from soku_ai.models.factory import build_model
from soku_ai.training.checkpoint import load_checkpoint


class TorchScriptInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        state_continuous: torch.Tensor,
        state_categorical: torch.Tensor,
        history_numerical: torch.Tensor,
        history_categorical: torch.Tensor,
        history_mask: torch.Tensor,
        self_object_numerical: torch.Tensor,
        self_object_categorical: torch.Tensor,
        self_object_mask: torch.Tensor,
        opponent_object_numerical: torch.Tensor,
        opponent_object_categorical: torch.Tensor,
        opponent_object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observation = {
            "state_continuous": state_continuous,
            "state_categorical": state_categorical,
            "history_numerical": history_numerical,
            "history_categorical": history_categorical,
            "history_mask": history_mask,
            "self_object_numerical": self_object_numerical,
            "self_object_categorical": self_object_categorical,
            "self_object_mask": self_object_mask,
            "opponent_object_numerical": opponent_object_numerical,
            "opponent_object_categorical": opponent_object_categorical,
            "opponent_object_mask": opponent_object_mask,
        }
        q_values = self.model(observation)
        return q_values, q_values.argmax(dim=1)


def example_inputs(config: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    history_len = int(config["data"]["history_len"])
    max_objects = int(config["data"]["max_objects_per_side"])
    shape = SHAPE if config["model"]["model_type"] == MODEL_TYPE else OBSERVATION_SHAPE
    return (
        torch.zeros(1, shape.state_continuous, device=device),
        torch.zeros(1, shape.state_categorical, dtype=torch.long, device=device),
        torch.zeros(1, history_len, shape.history_numerical, device=device),
        torch.full(
            (1, history_len, shape.history_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
            device=device,
        ),
        torch.zeros(1, history_len, dtype=torch.bool, device=device),
        torch.zeros(1, max_objects, shape.object_numerical, device=device),
        torch.full(
            (1, max_objects, shape.object_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
            device=device,
        ),
        torch.zeros(1, max_objects, dtype=torch.bool, device=device),
        torch.zeros(1, max_objects, shape.object_numerical, device=device),
        torch.full(
            (1, max_objects, shape.object_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
            device=device,
        ),
        torch.zeros(1, max_objects, dtype=torch.bool, device=device),
    )


def export_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).to(device).eval()
    load_checkpoint(checkpoint_path, online_network=model, map_location=device)
    wrapper = TorchScriptInferenceWrapper(model).eval()
    inputs = example_inputs(config, torch.device(device))
    traced = torch.jit.trace(wrapper, inputs, strict=False)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(destination))
    return destination
