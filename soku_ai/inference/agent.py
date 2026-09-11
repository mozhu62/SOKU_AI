from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from soku_ai.data.action_space import decode_action
from soku_ai.models.factory import build_model
from soku_ai.training.checkpoint import load_checkpoint
from soku_ai.training.utils import move_to_device, resolve_device


class SokuDQNAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        normalization: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.normalization = normalization
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.rng = np.random.default_rng(int(config["training"]["seed"]))

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> "SokuDQNAgent":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        resolved_device = resolve_device(device)
        model = build_model(config).to(resolved_device)
        load_checkpoint(
            checkpoint_path,
            online_network=model,
            map_location=resolved_device,
        )
        metadata_keys = (
            "checkpoint_kind",
            "training_step",
            "epoch",
            "model_architecture_version",
            "observation_schema_version",
            "preprocessing_version",
            "provenance",
        )
        metadata = {key: checkpoint.get(key) for key in metadata_keys}
        metadata["checkpoint_path"] = str(Path(checkpoint_path).resolve())
        return cls(
            model,
            config,
            resolved_device,
            checkpoint.get("normalization"),
            metadata,
        )

    @torch.no_grad()
    def predict_q(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        prepared: dict[str, torch.Tensor] = {}
        for name, value in observation.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            expected_rank = 2 if name.startswith("state_") else 3
            if name.endswith("_mask"):
                expected_rank = 2
            if tensor.ndim == expected_rank - 1:
                tensor = tensor.unsqueeze(0)
            prepared[name] = tensor
        return self.model(move_to_device(prepared, self.device))

    def select_action(
        self,
        observation: Mapping[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> dict[str, Any]:
        if self.rng.random() < float(epsilon):
            action_id = int(self.rng.integers(self.config["model"]["num_actions"]))
        else:
            action_id = int(self.predict_q(observation).argmax(dim=1).item())
        return {"action_id": action_id, **decode_action(action_id).to_dict()}
