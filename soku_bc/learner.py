from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models import BCNetwork
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


def classification_parts(logits, labels, mask, label_smoothing=0.0):
    if logits.shape[:-1] != labels.shape or mask.shape != labels.shape or logits.shape[-1] != ACTION_COUNT:
        raise ValueError("BC logits、标签与有效 mask 形状不一致")
    if mask.dtype != torch.bool or labels.dtype != torch.long:
        raise ValueError("有效 mask 必须为 bool，Joint Action 标签必须为 int64")
    # 先筛选再计算 CE；padding 和 burn-in 均不参与损失、命中率或梯度。
    valid_logits, valid_labels = logits.float()[mask], labels[mask]
    if not len(valid_labels):
        raise ValueError("BC 批次没有有效动作标签")
    loss = F.cross_entropy(valid_logits, valid_labels, label_smoothing=label_smoothing)
    return loss, {"logits": valid_logits, "labels": valid_labels}


def classification_metrics(parts):
    logits, labels = parts["logits"].detach(), parts["labels"]
    log_prob = logits.log_softmax(-1)
    probability = log_prob.exp()
    confidence, prediction = probability.max(-1)
    correct = prediction == labels
    expert_log_prob = log_prob.gather(-1, labels[:, None]).squeeze(-1)
    entropy = -(probability * log_prob).sum(-1)
    keys = ("nll", "joint_accuracy", "joint_top5", "direction_accuracy", "combat_accuracy",
            "card_accuracy", "expert_probability", "confidence_mean", "entropy", "normalized_entropy",
            "neutral_data_fraction", "neutral_pred_fraction", "logit_abs_max")
    scalars = torch.stack((-expert_log_prob.mean(), correct.float().mean(),
                           (logits.topk(5, dim=-1).indices == labels[:, None]).any(-1).float().mean(),
                           (prediction // 48 == labels // 48).float().mean(),
                           ((prediction % 48) // 3 == (labels % 48) // 3).float().mean(),
                           (prediction % 3 == labels % 3).float().mean(),
                           expert_log_prob.exp().mean(), confidence.mean(), entropy.mean(),
                           entropy.mean() / math.log(ACTION_COUNT),
                           (labels == NEUTRAL_ACTION_ID).float().mean(),
                           (prediction == NEUTRAL_ACTION_ID).float().mean(), logits.abs().max())).cpu().tolist()
    result = dict(zip(keys, scalars))
    result.update(samples=len(labels),
                  joint_data=torch.bincount(labels, minlength=ACTION_COUNT).cpu().tolist(),
                  joint_pred=torch.bincount(prediction, minlength=ACTION_COUNT).cpu().tolist(),
                  joint_correct=torch.bincount(labels[correct], minlength=ACTION_COUNT).cpu().tolist())
    result["joint_data_top"] = frequency_rows(result["joint_data"])
    result["joint_pred_top"] = frequency_rows(result["joint_pred"])
    return result


def aggregate_metrics(rows):
    """按有效帧加权合并，不让片段 padding 或小批次改变验证口径。"""
    count = sum(row["samples"] for row in rows)
    if count <= 0:
        raise ValueError("验证没有有效样本")
    result = {}
    for key in rows[0]:
        if key in ("samples", "joint_data_top", "joint_pred_top"):
            continue
        if key in ("joint_data", "joint_pred", "joint_correct"):
            result[key] = np.sum([row[key] for row in rows], axis=0, dtype=np.int64).tolist()
        elif key == "logit_abs_max":
            result[key] = max(row[key] for row in rows)
        else:
            result[key] = sum(row[key] * row["samples"] for row in rows) / count
    data = np.asarray(result["joint_data"])
    correct = np.asarray(result["joint_correct"])
    represented = data > 0
    result.update(samples=count, represented_actions=int(represented.sum()),
                  macro_recall=float(np.mean(correct[represented] / data[represented])),
                  joint_data_top=frequency_rows(result["joint_data"]),
                  joint_pred_top=frequency_rows(result["joint_pred"]))
    return result


class Learner:
    """单网络行为克隆：REP 完整按键为分类标签，不读取奖励，不建立目标网络。"""

    def __init__(self, config):
        self.config = config
        self.device = device_for(config["training"]["device"])
        torch.set_num_threads(config["training"]["cpu_threads"])
        torch.manual_seed(config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.model = BCNetwork(config["model"]).to(self.device)
        cfg = config["training"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        self.amp = cfg["amp"] and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.apply_settings(config)

    def apply_settings(self, config):
        self.config = config
        cfg = config["training"]
        for name, module in self.model.module_groups().items():
            module.requires_grad_(name not in cfg["frozen_modules"])
            for parameter in module.parameters():
                parameter.grad = None
        for group in self.optimizer.param_groups:
            group.update(lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    def losses(self, batch):
        cfg = self.config["training"]
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(batch["observation"], cfg["burn_in"], batch["burn_lengths"])
        # CE 在 float32 中执行；网络输出原始 logits，不预先做 softmax。
        return classification_parts(logits, batch["joint_action_id"], batch["mask"], cfg["label_smoothing"])

    def train_batch(self, raw_batch, diagnostics=False):
        self.model.train()
        started = time.perf_counter()
        batch = tensor_batch(raw_batch, self.device)
        before = {name: [p.detach().clone() for p in module.parameters()]
                  for name, module in self.model.module_groups().items()} if diagnostics else {}
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = self.losses(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("BC 损失出现 NaN/Inf，已中止当前更新")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradients = {}
        if diagnostics:
            for name, module in self.model.module_groups().items():
                gradients[name] = float(torch.stack([p.grad.detach().float().square().sum() for p in module.parameters()
                                                     if p.grad is not None] or [loss.new_zeros(())]).sum().sqrt())
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])
        previous_scale = self.scaler.get_scale()
        if not self.amp and not torch.isfinite(grad):
            raise FloatingPointError("梯度出现 NaN/Inf，已阻止优化器更新")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < previous_scale
        result = {"optimizer_skipped": skipped, "samples": len(parts["labels"])}
        # 完整动作频率、概率与模块权重差只在记录步计算，减少 GPU/CPU 同步。
        if diagnostics:
            result.update(classification_metrics(parts), loss=float(loss.detach()),
                          grad_norm=float(grad) if torch.isfinite(grad) else None,
                          module_gradients={name: value if math.isfinite(value) else None for name, value in gradients.items()})
            result["module_changes"] = {name: float(torch.stack([(p.detach() - old).float().square().sum()
                                                                 for p, old in zip(module.parameters(), before[name])]).sum().sqrt())
                                        for name, module in self.model.module_groups().items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result["optimization_seconds"] = time.perf_counter() - started
        return result

    @torch.no_grad()
    def validate_batch(self, raw_batch):
        self.model.eval()
        loss, parts = self.losses(tensor_batch(raw_batch, self.device))
        if not torch.isfinite(loss):
            raise FloatingPointError("验证损失出现 NaN/Inf")
        return {**classification_metrics(parts), "loss": float(loss)}
