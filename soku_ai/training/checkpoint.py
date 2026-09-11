from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from soku_ai.data.action_space import action_mapping_metadata
from soku_ai.data.normalization import NormalizationStats
from soku_ai.data.schemas import SCHEMA_VERSION
from soku_ai.data.split import load_split_manifest

from .utils import capture_rng_state, restore_rng_state


CHECKPOINT_VERSION = 1


def build_checkpoint(
    *,
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    normalization: NormalizationStats,
    step: int,
    epoch: int,
    seed: int,
    best_metric: float,
    scaler: Any | None = None,
    replay_buffer_state: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_split_manifest(config["data"]["split_manifest"])
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": "resume",
        "model_architecture_version": getattr(
            online_network, "architecture_version", "unknown"
        ),
        "observation_schema_version": getattr(online_network, "observation_schema_version", SCHEMA_VERSION),
        "preprocessing_version": config["data"]["preprocessing_version"],
        "model_state_dict": online_network.state_dict(),
        "target_model_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "normalization": normalization.to_dict(),
        "action_mapping": action_mapping_metadata(),
        "training_step": int(step),
        "epoch": int(epoch),
        "random_seed": int(seed),
        "dataset_split_manifest_hash": manifest["sha256"],
        "best_metric": float(best_metric),
        "rng_state": capture_rng_state(),
    }
    if scaler is not None:
        checkpoint["amp_scaler_state_dict"] = scaler.state_dict()
    if replay_buffer_state is not None:
        checkpoint["replay_buffer_state"] = replay_buffer_state
    if provenance:
        checkpoint["provenance"] = dict(provenance)
    return checkpoint


def build_model_snapshot(
    *,
    online_network: nn.Module,
    config: dict,
    normalization: NormalizationStats,
    step: int,
    epoch: int,
    seed: int,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成用于实战对比和推理导出的轻量模型版本，不包含续训状态。"""
    manifest = load_split_manifest(config["data"]["split_manifest"])
    snapshot = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_kind": "model_snapshot",
        "model_architecture_version": getattr(
            online_network, "architecture_version", "unknown"
        ),
        "observation_schema_version": getattr(online_network, "observation_schema_version", SCHEMA_VERSION),
        "preprocessing_version": config["data"]["preprocessing_version"],
        "model_state_dict": online_network.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "normalization": normalization.to_dict(),
        "action_mapping": action_mapping_metadata(),
        "training_step": int(step),
        "epoch": int(epoch),
        "random_seed": int(seed),
        "dataset_split_manifest_hash": manifest["sha256"],
    }
    if provenance:
        snapshot["provenance"] = dict(provenance)
    return snapshot


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    online_network: nn.Module,
    target_network: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"不支持的 checkpoint_version: {checkpoint.get('checkpoint_version')}")
    if (checkpoint.get("model_architecture_version") != getattr(online_network, "architecture_version", "unknown")
            or checkpoint.get("observation_schema_version") != getattr(online_network, "observation_schema_version", SCHEMA_VERSION)):
        raise ValueError("checkpoint 输入/网络版本不兼容：旧 DQfD 与 resources_v4 不可直接迁移，请去掉 --resume/--init-from 从零训练")
    if checkpoint.get("checkpoint_kind", "resume") == "model_snapshot" and (
        target_network is not None or optimizer is not None
    ):
        raise ValueError("轻量模型版本只能用于推理，续训请使用 last.pt")
    online_network.load_state_dict(checkpoint["model_state_dict"])
    if target_network is not None:
        target_network.load_state_dict(checkpoint["target_model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "amp_scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
