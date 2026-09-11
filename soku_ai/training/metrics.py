from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def q_metrics(q_values: torch.Tensor, expert_actions: torch.Tensor) -> dict[str, float]:
    expert_actions = expert_actions.long()
    expert_q = q_values.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
    expert_mask = torch.nn.functional.one_hot(
        expert_actions, num_classes=q_values.shape[1]
    ).bool()
    other_q = q_values.masked_fill(expert_mask, 0.0).sum(dim=1) / max(q_values.shape[1] - 1, 1)
    top1 = q_values.argmax(dim=1).eq(expert_actions).float().mean()
    top5 = q_values.topk(min(5, q_values.shape[1]), dim=1).indices.eq(
        expert_actions.unsqueeze(1)
    ).any(dim=1).float().mean()
    return {
        "expert_top1": float(top1.item()),
        "expert_top5": float(top5.item()),
        "q_expert": float(expert_q.mean().item()),
        "q_other": float(other_q.mean().item()),
        "q_mean": float(q_values.mean().item()),
        "q_max": float(q_values.max().item()),
        "q_min": float(q_values.min().item()),
        "q_abs_mean": float(q_values.abs().mean().item()),
        "q_abs_max": float(q_values.abs().max().item()),
    }


def no_op_metrics(
    q_values: torch.Tensor,
    expert_actions: torch.Tensor,
    self_actions: torch.Tensor,
    no_op_action_id: int = 0,
) -> dict[str, float]:
    predicted = q_values.argmax(dim=1)
    expert_actions = expert_actions.long()
    self_actions = self_actions.long()
    no_op_id = int(no_op_action_id)
    non_no_op = expert_actions != no_op_id
    idle = self_actions == 0
    other_q = q_values.clone()
    other_q[:, no_op_id] = torch.finfo(other_q.dtype).min
    result = {
        "predicted_no_op_rate": float(predicted.eq(no_op_id).float().mean().item()),
        "expert_no_op_rate": float(expert_actions.eq(no_op_id).float().mean().item()),
        "q_no_op_advantage": float(
            (q_values[:, no_op_id] - other_q.max(dim=1).values).mean().item()
        ),
    }
    if non_no_op.any():
        result["non_no_op_top1"] = float(
            predicted[non_no_op].eq(expert_actions[non_no_op]).float().mean().item()
        )
    if idle.any():
        result["idle_predicted_no_op_rate"] = float(
            predicted[idle].eq(no_op_id).float().mean().item()
        )
        result["idle_q_no_op_advantage"] = float(
            (
                q_values[idle, no_op_id]
                - other_q[idle].max(dim=1).values
            ).mean().item()
        )
    return result


def action_style_metrics(
    q_values: torch.Tensor,
    expert_actions: torch.Tensor,
) -> dict[str, float]:
    predicted = q_values.argmax(dim=1).long()
    expert = expert_actions.long()

    def rates(prefix: str, actions: torch.Tensor) -> dict[str, float]:
        horizontal = actions // 48
        buttons = actions % 16
        return {
            f"{prefix}_forward_rate": float(horizontal.eq(1).float().mean().item()),
            f"{prefix}_backward_rate": float(horizontal.eq(2).float().mean().item()),
            f"{prefix}_attack_abc_rate": float(
                torch.bitwise_and(buttons, 0b1110).ne(0).float().mean().item()
            ),
            f"{prefix}_dash_d_rate": float(
                torch.bitwise_and(buttons, 0b0001).ne(0).float().mean().item()
            ),
        }

    return {**rates("predicted", predicted), **rates("expert", expert)}


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    weights: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, metrics: dict[str, float], weight: float = 1.0) -> None:
        for key, value in metrics.items():
            if np.isfinite(value):
                self.sums[key] += float(value) * weight
                self.weights[key] += weight

    def result(self) -> dict[str, float]:
        return {
            key: self.sums[key] / self.weights[key]
            for key in self.sums
            if self.weights[key] > 0
        }


def object_count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 4:
        return "1-4"
    if value <= 16:
        return "5-16"
    if value <= 32:
        return "17-32"
    return "33+"
