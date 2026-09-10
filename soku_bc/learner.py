from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .models import BCNetwork
from .config import active_modules
from .action_space import ACTION_COUNT, NEUTRAL_ACTION_ID, frequency_rows
from .action_diagnostics import BATCH_COUNT_KEYS, BATCH_RATE_KEYS, batch_history_rates


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
    # 先筛选再计算 CE；padding/前导帧不产生独立标签损失或命中率。
    # 31 帧前导不产生独立 CE，但可接收后续监督帧经 TCN 传回的梯度。
    valid_logits, valid_labels = logits.float()[mask], labels[mask]
    if not len(valid_labels):
        raise ValueError("BC 批次没有有效动作标签")
    loss = F.cross_entropy(valid_logits, valid_labels, label_smoothing=label_smoothing)
    return loss, {"logits": valid_logits, "labels": valid_labels}


@torch.no_grad()
def classification_metrics(parts, previous_joint_action_id=None):
    logits, labels = parts["logits"].detach(), parts["labels"]
    log_prob = logits.log_softmax(-1)
    probability = log_prob.exp()
    confidence, prediction = probability.max(-1)
    correct = prediction == labels
    top5_correct = (logits.topk(5, dim=-1).indices == labels[:, None]).any(-1)
    expert_log_prob = log_prob.gather(-1, labels[:, None]).squeeze(-1)
    entropy = -(probability * log_prob).sum(-1)
    keys = ("nll", "joint_accuracy", "joint_top5", "direction_accuracy", "combat_accuracy",
            "card_accuracy", "expert_probability", "confidence_mean", "entropy", "normalized_entropy",
            "neutral_data_fraction", "neutral_pred_fraction", "logit_abs_max")
    scalars = torch.stack((-expert_log_prob.mean(), correct.float().mean(),
                           top5_correct.float().mean(),
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
    if previous_joint_action_id is not None:
        previous = previous_joint_action_id.detach()
        if previous.shape != labels.shape:
            raise ValueError("动作历史诊断没有与有效标签逐帧对齐")
        eligible = (previous >= 0) & (previous < ACTION_COUNT)
        same = previous == labels
        changed = eligible & ~same
        counts = torch.stack((eligible.sum(), (eligible & same).sum(), changed.sum(),
                              (changed & correct).sum(), (changed & top5_correct).sum(),
                              (eligible & correct).sum())).cpu().tolist()
        result.update(dict(zip(BATCH_COUNT_KEYS, counts)))
        result.update(batch_history_rates(result, len(labels)))
    return result


def diagnostic_previous_actions(batch, burn_in):
    # 从同一批 observation 取真实历史：跳过 burn-in，应用与 CE 完全相同的 mask。
    # 若批次已镜像，这里读取的当前标签和历史动作自然使用同一增强坐标系。
    mask = batch["mask"]
    previous = batch["observation"]["previous_joint_action_id"][:, burn_in:burn_in + mask.shape[1]]
    if previous.shape != mask.shape:
        raise ValueError("动作历史长度与监督片段不一致")
    return previous[mask]


def aggregate_metrics(rows):
    """按有效帧加权合并，不让片段 padding 或小批次改变验证口径。"""
    count = sum(row["samples"] for row in rows)
    if count <= 0:
        raise ValueError("验证没有有效样本")
    result = {}
    for key in rows[0]:
        if key in ("samples", "joint_data_top", "joint_pred_top", *BATCH_COUNT_KEYS, *BATCH_RATE_KEYS):
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
    # 各批切换帧数量不同，必须合并分子/分母；不可按总帧数平均切换准确率。
    if all(all(key in row for key in BATCH_COUNT_KEYS) for row in rows):
        result.update({key: sum(row[key] for row in rows) for key in BATCH_COUNT_KEYS})
        result.update(batch_history_rates(result, count))
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
        active = set(active_modules(config["model"]))
        for name, module in self.model.module_groups().items():
            module.requires_grad_(name in active and name not in cfg["frozen_modules"])
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
            result.update(classification_metrics(parts, diagnostic_previous_actions(batch, self.config["training"]["burn_in"])),
                          loss=float(loss.detach()),
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
        batch = tensor_batch(raw_batch, self.device)
        loss, parts = self.losses(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("验证损失出现 NaN/Inf")
        return {**classification_metrics(parts, diagnostic_previous_actions(batch, self.config["training"]["burn_in"])),
                "loss": float(loss)}
