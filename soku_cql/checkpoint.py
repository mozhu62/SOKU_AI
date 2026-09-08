from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from .config import NETWORK_VERSION
from .action_space import ACTION_SCHEMA
from .schema import policy_input_manifest
from .n_step import target_spec


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"未找到 CQL checkpoint：{path}；从零训练请去掉 --resume")
    package = torch.load(path, map_location="cpu", weights_only=True)
    version = package.get("network_version")
    if version != NETWORK_VERSION:
        raise ValueError("CQL checkpoint schema 不兼容：旧动作语义、观测输入及 Q 头均已替换。"
                         "joint432 不允许部分加载；请去掉 --resume，从随机初始化开始并使用新输出目录")
    if package.get("spec", {}).get("network_version") != version:
        raise ValueError("checkpoint 顶层网络版本与 spec 不一致")
    if (package["spec"].get("action_schema") != ACTION_SCHEMA or
            package["spec"].get("inputs") != policy_input_manifest()):
        raise ValueError("checkpoint action/observation schema 不兼容，不允许部分加载")
    # 仅补齐内存中的历史配置，不改磁盘文件或权重；显式 YAML 才切换到新的 N。
    package["config"]["training"].setdefault("n_step", 1)
    # 历史模型没有这个辅助项；仅 --resume 时保持关闭，显式配置/工作台应用才启用。
    package["config"]["training"].setdefault("expert_imitation_weight", 0.0)
    saved_target = package.get("td_target")
    if saved_target is not None and saved_target != target_spec(package["config"]["training"]):
        raise ValueError("checkpoint 的 TD 目标版本或 N/gamma 与保存配置不一致")
    return package


def save(path, learner, config, split, normalization, step, updates, samples, best, stage):
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu(item) for item in value)
        return value
    package = {"network_version": learner.online.spec["network_version"], "spec": learner.online.spec,
               "online": cpu(learner.online.state_dict()), "target": cpu(learner.target.state_dict()),
               "optimizer": cpu(learner.optimizer.state_dict()), "scaler": learner.scaler.state_dict(),
               "config": config, "td_target": target_spec(config["training"]),
               "split_hash": split["sha256"], "normalization": normalization,
               "step": step, "updates": updates, "samples": samples, "best": best, "stage": stage,
               "rng_cpu": torch.get_rng_state(),
               "rng_cuda": torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(package, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore(package, learner, split):
    if package["spec"] != learner.online.spec or package["split_hash"] != split["sha256"]:
        raise ValueError("checkpoint 的网络/动作/输入结构或固定数据划分不兼容")
    learner.online.load_state_dict(package["online"], strict=True)
    learner.target.load_state_dict(package["target"], strict=True)
    learner.optimizer.load_state_dict(package["optimizer"])
    learner.scaler.load_state_dict(package["scaler"])
    learner.apply_settings(learner.config)
    torch.set_rng_state(package["rng_cpu"].cpu().to(torch.uint8))
    if learner.device.type == "cuda" and len(package["rng_cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([value.cpu().to(torch.uint8) for value in package["rng_cuda"]])
