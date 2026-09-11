from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn


def concatenate_observations(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("至少需要一个 Observation 才能拼接")
    keys = tuple(observations[0].keys())
    if any(tuple(observation.keys()) != keys for observation in observations[1:]):
        raise ValueError("待拼接 Observation 的字段不一致")
    return {
        key: torch.cat([observation[key] for observation in observations], dim=0)
        for key in keys
    }


@torch.no_grad()
def double_dqn_next_q(
    online_network: nn.Module,
    target_network: nn.Module,
    next_observation: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    was_training = online_network.training
    online_network.eval()
    try:
        next_actions = online_network(next_observation).argmax(dim=1, keepdim=True)
    finally:
        online_network.train(was_training)
    return target_network(next_observation).gather(1, next_actions).squeeze(1)


@torch.no_grad()
def double_dqn_target(
    online_network: nn.Module,
    target_network: nn.Module,
    next_observation: Mapping[str, torch.Tensor],
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discounts: torch.Tensor,
) -> torch.Tensor:
    next_q = double_dqn_next_q(online_network, target_network, next_observation)
    return rewards + (~dones.bool()).to(rewards.dtype) * discounts * next_q
