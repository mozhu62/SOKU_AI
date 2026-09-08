from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models import CQLNetwork, conservative_gap, selected_q
from .action_space import ACTION_COUNT, NEUTRAL_ACTION_ID, frequency_rows


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("当前离线训练支持 CPU 或 PyTorch CUDA/ROCm 设备")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前 PyTorch 没有可用的 CUDA/ROCm 设备")
    return device


def prepare_cpu_batch(batch, pin_memory=False):
    def convert(value, cpu=False):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        return tensor.pin_memory() if pin_memory and not cpu else tensor
    return {key: ({k: convert(v) for k, v in value.items()} if key == "observation"
                  else convert(value, cpu=key == "burn_lengths")) for key, value in batch.items()}


def tensor_batch(batch, device):
    if not isinstance(batch["mask"], torch.Tensor):
        batch = prepare_cpu_batch(batch, device.type == "cuda")
    return {key: ({k: v.to(device, non_blocking=True) for k, v in value.items()} if key == "observation"
                  else value if key == "burn_lengths" else value.to(device, non_blocking=True))
            for key, value in batch.items()}


class Learner:
    """纯离线离散 CQL：Double DQN 目标 + 联合动作保守项，无 PPO rollout/ratio/GAE。"""

    def __init__(self, config):
        self.config = config
        self.device = device_for(config["training"]["device"])
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.manual_seed(config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.online = CQLNetwork(config["model"]).to(self.device)
        self.target = copy.deepcopy(self.online).eval().requires_grad_(False)
        cfg = config["training"]
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        self.amp = cfg["amp"] and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.apply_settings(config)

    def apply_settings(self, config):
        self.config = config
        cfg = config["training"]
        for name, module in self.online.module_groups().items():
            module.requires_grad_(name not in cfg["frozen_modules"])
            for parameter in module.parameters():
                parameter.grad = None
        for group in self.optimizer.param_groups:
            group.update(lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    def losses(self, batch):
        cfg = self.config["training"]
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            joint_q = self.online(batch["observation"], cfg["burn_in"], batch["burn_lengths"])
            with torch.no_grad():
                target_q = self.target(batch["observation"], cfg["burn_in"], batch["burn_lengths"])
        # logsumexp、Bellman 目标及损失使用 float32，避免 AMP 下指数和回报溢出。
        joint_q = joint_q.float()
        length = batch["joint_action_id"].shape[1]
        current_q = joint_q[:, :length]
        data_q = selected_q(current_q, batch["joint_action_id"])
        with torch.no_grad():
            # Double DQN 在 t+k 而非 t+1 选动作；在线与目标 GRU 都已经读过中间真实状态。
            indices = batch["bootstrap_indices"].unsqueeze(-1).expand(-1, -1, ACTION_COUNT)
            online_next = joint_q.gather(1, indices)
            target_next = target_q.float().gather(1, indices)
            next_q = selected_q(target_next, online_next.argmax(-1))
            discounts = batch["bootstrap_discounts"]
            # 真终局与 padding 不使用 Q，避免 0 × 无效 Q 污染本来不应 bootstrap 的目标。
            target = batch["n_step_returns"] + discounts * torch.where(discounts > 0, next_q, 0.0)
        mask = batch["mask"]
        q, y = data_q[mask], target[mask]
        gap = conservative_gap(current_q, data_q, cfg["cql_temperature"])[mask]
        td_loss = F.smooth_l1_loss(q, y)
        loss = td_loss + cfg["cql_alpha"] * gap.mean()
        return loss, {"q": q, "target": y, "gap": gap, "td_loss": td_loss,
                      "joint_q": current_q[mask], "joint_labels": batch["joint_action_id"][mask],
                      "n_step_steps": batch["n_step_steps"][mask],
                      "n_step_full": (batch["n_step_steps"][mask] == cfg["n_step"]),
                      "bootstrap_active": (batch["bootstrap_discounts"][mask] > 0)}

    @staticmethod
    def metrics(parts):
        q, y = parts["q"].detach(), parts["target"].detach()
        errors = q - y
        joint = parts["joint_q"].detach()
        q_max, prediction = joint.max(-1)
        labels = parts["joint_labels"]
        variance = y.var(unbiased=False)
        scalars = torch.stack((parts["td_loss"].detach(), parts["gap"].detach().mean(), errors.square().mean(),
                               errors.abs().mean(), q.mean(), q.std(unbiased=False), joint.abs().max(),
                               y.mean(), y.std(unbiased=False), variance, errors.var(unbiased=False),
                               (prediction == labels).float().mean(), q_max.mean(), q_max.std(unbiased=False),
                               (q_max - q).mean(), (labels == NEUTRAL_ACTION_ID).float().mean(),
                               (prediction == NEUTRAL_ACTION_ID).float().mean(),
                               parts["n_step_steps"].float().mean(), parts["n_step_full"].float().mean(),
                               parts["bootstrap_active"].float().mean())).cpu().tolist()
        keys = ("td_loss", "cql_gap", "td_mse", "td_mae", "q_data_mean", "q_data_std", "q_abs_max", "target_mean",
                "target_std", "target_variance", "error_variance", "joint_accuracy", "q_max_mean", "q_max_std",
                "q_max_minus_q_data", "neutral_data_fraction", "neutral_pred_fraction",
                "n_step_mean", "n_step_full_fraction", "bootstrap_fraction")
        result = dict(zip(keys, scalars))
        result["ev"] = 1 - result["error_variance"] / result["target_variance"] if result["target_variance"] > 1e-8 else None
        result["samples"] = len(q)
        result["joint_pred"] = torch.bincount(prediction, minlength=ACTION_COUNT).cpu().tolist()
        result["joint_data"] = torch.bincount(labels, minlength=ACTION_COUNT).cpu().tolist()
        result["joint_data_top"] = frequency_rows(result["joint_data"])
        result["joint_pred_top"] = frequency_rows(result["joint_pred"])
        return result

    def train_batch(self, raw_batch, diagnostics=False):
        self.online.train()
        started = time.perf_counter()
        batch = tensor_batch(raw_batch, self.device)
        before = {name: [p.detach().clone() for p in module.parameters()]
                  for name, module in self.online.module_groups().items()} if diagnostics else {}
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = self.losses(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("CQL 损失出现 NaN/Inf，已中止当前更新")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradients = {}
        if diagnostics:
            for name, module in self.online.module_groups().items():
                gradients[name] = float(torch.stack([p.grad.detach().float().square().sum() for p in module.parameters()
                                                     if p.grad is not None] or [loss.new_zeros(())]).sum().sqrt())
        grad = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config["training"]["max_grad_norm"])
        previous_scale = self.scaler.get_scale()
        if not self.amp and not torch.isfinite(grad):
            raise FloatingPointError("梯度出现 NaN/Inf，已阻止优化器更新")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < previous_scale
        if not skipped:
            with torch.no_grad():
                for online, target in zip(self.online.parameters(), self.target.parameters(), strict=True):
                    target.lerp_(online, self.config["training"]["target_tau"])
        # 只在记录步同步诊断；每步不向 GUI 搬运整套权重或完整轨迹。
        result = {"optimizer_skipped": skipped, "samples": int(raw_batch["mask"].sum().item())}
        if diagnostics:
            result.update(self.metrics(parts), loss=float(loss.detach()),
                          grad_norm=float(grad) if torch.isfinite(grad) else None,
                          module_gradients={name: value if math.isfinite(value) else None for name, value in gradients.items()})
            result["module_changes"] = {name: float(torch.stack([(p.detach() - old).float().square().sum()
                                                                 for p, old in zip(module.parameters(), before[name])]).sum().sqrt())
                                        for name, module in self.online.module_groups().items()}
        if self.device.type == "cuda":
            # 准确计入异步 GPU 工作耗时；loss 有限性检查本身已经建立了同步点。
            torch.cuda.synchronize(self.device)
        result["optimization_seconds"] = time.perf_counter() - started
        return result

    @torch.no_grad()
    def validate_batch(self, raw_batch):
        self.online.eval()
        loss, parts = self.losses(tensor_batch(raw_batch, self.device))
        if not torch.isfinite(loss):
            raise FloatingPointError("验证损失出现 NaN/Inf")
        return {**self.metrics(parts), "loss": float(loss)}
