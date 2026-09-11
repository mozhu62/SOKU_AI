"""PALR 历史表示正则；不参与实战推理，不包含可训练参数。

HSCIC 对应 PALR (NeurIPS 2023) 式 (7)，等价移植自
https://github.com/KAIST-AILab/palr/blob/main/core/hscic.py
原实现 MIT 授权见 docs/licenses/PALR-MIT.txt。
"""
from __future__ import annotations

import math

import torch

from .action_space import ACTION_COUNT
from .keyframes import build_changepoint_mask


def categorical_kernel(actions):
    if actions.ndim != 1:
        raise ValueError("categorical kernel 需要一维离散动作标签")
    return (actions[:, None] == actions[None, :]).float()


def feature_rbf_kernel(features):
    """K=exp(-距离平方/带宽)，带宽为非零非对角距离平方的 detached 中位数。"""
    # 共同缩放在中位数带宽下不改变距离比，避免极大有限特征在平方时溢出。
    x = features.float()
    scale = x.detach().abs().amax().clamp_min(1.0)
    x = x / scale
    x = x - x.mean(0, keepdim=True)
    squared = x.square().sum(-1)
    distance = (squared[:, None] + squared[None, :] - 2 * (x @ x.T)).clamp_min(0)
    diagonal = torch.eye(len(x), device=x.device, dtype=torch.bool)
    distance = distance.masked_fill(diagonal, 0)
    positive = distance.detach()[~diagonal & (distance.detach() > 0)]
    bandwidth = positive.median() if positive.numel() else x.new_tensor(1.0)
    bandwidth = bandwidth.clamp_min(1e-12).detach()
    return torch.exp(-(distance / bandwidth).clamp_max(80))


def compute_hscic(features, previous_actions, current_actions, regularization=0.001):
    """HSCIC(phi, 上一真实专家动作 | 当前专家动作)，返回可微 float32 标量。"""
    if features.ndim != 2 or previous_actions.shape != (len(features),) or current_actions.shape != (len(features),):
        raise ValueError("HSCIC 输入必须为 [N,D]、[N]、[N]")
    if not math.isfinite(regularization) or regularization < 1e-6:
        raise ValueError("HSCIC regularization 必须为不小于 1e-6 的有限数值")
    if len(features) < 2:
        return features.float().sum() * 0
    # 外层启用 AMP 时也不能让核矩阵和线性求解降为半精度。
    with torch.autocast(device_type=features.device.type, enabled=False):
        kx = feature_rbf_kernel(features)
        ky = categorical_kernel(previous_actions.to(features.device))
        kz = categorical_kernel(current_actions.to(features.device))
        n = len(features)
        ridge = kz + (n * regularization) * torch.eye(n, device=kx.device, dtype=kx.dtype)
        a = torch.linalg.solve(ridge, kz)
        # A=(Kz+nλI)^(-1)Kz；以下三个迹项与论文式 (7) 及官方实现相同。
        # 只改用离散动作核和稳健中位数特征带宽，不换成中心化 HSIC 或残差代理。
        kxa, kya = kx @ a, ky @ a
        term1 = (a * ((kx * ky) @ a)).sum()
        term2 = (a * kxa * kya).sum()
        term3 = ((a * kxa).sum(0) * (a * kya).sum(0)).sum()
        return ((term1 - 2 * term2 + term3) / n).clamp_min(0)


def sampled_palr(features, current_expert_actions, previous_expert_actions, valid_mask,
                 settings, *, generator=None):
    """只抽 PALR 帧；不会改动 BC 的标签、监督 mask 或采样轨迹。"""
    if features.shape[:-1] != current_expert_actions.shape:
        raise ValueError("TCN feature 与监督标签没有逐帧对齐")
    _, eligible = build_changepoint_mask(current_expert_actions, valid_mask, previous_expert_actions)
    eligible &= (current_expert_actions >= 0) & (current_expert_actions < ACTION_COUNT)
    indices = eligible.flatten().nonzero(as_tuple=True)[0]
    count = indices.numel()
    if count > settings["sample_size"]:
        # CPU torch RNG 已由 Learner seed 初始化并进入 checkpoint；验证传入独立 generator。
        order = torch.randperm(count, generator=generator, device="cpu")[:settings["sample_size"]]
        indices = indices[order.to(indices.device)]
    selected = features.reshape(-1, features.shape[-1])[indices]
    stats = {"palr_eligible_samples": count, "palr_sampled_samples": len(indices),
             "palr_skipped": len(indices) < 2,
             "palr_skip_reason": "too_few_continuous_expert_frames" if len(indices) < 2 else None}
    if len(indices) < 2:
        return selected.float().sum() * 0, stats
    loss = compute_hscic(selected, previous_expert_actions.flatten()[indices],
                        current_expert_actions.flatten()[indices], settings["regularization"])
    return loss, stats
