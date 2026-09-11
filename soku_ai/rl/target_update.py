from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    tau = float(tau)
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True
    ):
        target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


def update_target_if_needed(
    target: nn.Module,
    online: nn.Module,
    *,
    step: int,
    mode: str,
    interval: int,
    tau: float,
) -> bool:
    if mode == "hard":
        if step > 0 and step % int(interval) == 0:
            hard_update(target, online)
            return True
        return False
    if mode == "soft":
        soft_update(target, online, tau)
        return True
    raise ValueError(f"未知 Target 更新方式: {mode}")

