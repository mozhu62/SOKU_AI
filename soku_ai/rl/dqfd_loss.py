from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class DQfDLossOutput:
    total: torch.Tensor
    td1: torch.Tensor
    n_step: torch.Tensor
    margin: torch.Tensor
    idle_no_op: torch.Tensor
    l2: torch.Tensor
    td_errors: torch.Tensor


def large_margin_loss(
    q_values: torch.Tensor,
    expert_actions: torch.Tensor,
    is_demo: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    expert_actions = expert_actions.long()
    expert_q = q_values.gather(1, expert_actions.unsqueeze(1)).squeeze(1)
    margin_values = q_values + float(margin)
    margin_values = margin_values.scatter(1, expert_actions.unsqueeze(1), expert_q.unsqueeze(1))
    losses = (margin_values.max(dim=1).values - expert_q).clamp_min(0.0)
    return losses * is_demo.to(losses.dtype)


def parameter_l2(parameters: list[torch.Tensor]) -> torch.Tensor:
    if not parameters:
        return torch.tensor(0.0)
    squared = sum(parameter.float().pow(2).sum() for parameter in parameters)
    count = sum(parameter.numel() for parameter in parameters)
    return squared / max(count, 1)


def compute_dqfd_loss(
    *,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    td1_targets: torch.Tensor,
    n_step_targets: torch.Tensor,
    is_demo: torch.Tensor,
    importance_weights: torch.Tensor,
    online_network: nn.Module,
    config: dict,
    self_actions: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> DQfDLossOutput:
    chosen_q = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    td_errors = td1_targets.detach() - chosen_q
    td1_per_item = functional.smooth_l1_loss(chosen_q, td1_targets.detach(), reduction="none")
    n_step_per_item = functional.smooth_l1_loss(
        chosen_q, n_step_targets.detach(), reduction="none"
    )
    margin_per_item = large_margin_loss(
        q_values,
        actions,
        is_demo,
        float(config["margin"]),
    )
    no_op_action_id = int(config.get("no_op_action_id", 0))
    idle_state_action_id = int(config.get("idle_state_action_id", 0))
    no_op_margin_weight = float(config.get("no_op_margin_weight", 1.0))
    idle_no_op_margin_weight = float(
        config.get("idle_no_op_margin_weight", no_op_margin_weight)
    )
    if not 0.0 <= no_op_margin_weight <= 1.0 or not 0.0 <= idle_no_op_margin_weight <= 1.0:
        raise ValueError("动作 0 的 margin 权重必须位于 0..1")
    actions_long = actions.long()
    margin_action_weights = torch.where(
        actions_long == no_op_action_id,
        torch.full_like(margin_per_item, no_op_margin_weight),
        torch.ones_like(margin_per_item),
    )
    idle_mask = torch.zeros_like(actions_long, dtype=torch.bool)
    if self_actions is not None:
        idle_mask = self_actions.long() == idle_state_action_id
        idle_no_op_mask = idle_mask & (actions_long == no_op_action_id)
        margin_action_weights = torch.where(
            idle_no_op_mask,
            torch.full_like(margin_action_weights, idle_no_op_margin_weight),
            margin_action_weights,
        )
    margin_per_item = margin_per_item * margin_action_weights

    transition_weights = torch.ones_like(margin_per_item)
    if self_actions is not None:
        idle_no_op_weight = float(config.get("idle_no_op_transition_weight", 1.0))
        idle_non_no_op_weight = float(
            config.get("idle_non_no_op_transition_weight", 1.0)
        )
        if idle_no_op_weight <= 0.0 or idle_non_no_op_weight <= 0.0:
            raise ValueError("空闲状态 Transition 权重必须大于 0")
        transition_weights = torch.where(
            idle_mask & (actions_long == no_op_action_id),
            torch.full_like(transition_weights, idle_no_op_weight),
            transition_weights,
        )
        transition_weights = torch.where(
            idle_mask & (actions_long != no_op_action_id),
            torch.full_like(transition_weights, idle_non_no_op_weight),
            transition_weights,
        )
    if sample_weights is not None:
        transition_weights = transition_weights * sample_weights.to(q_values.dtype)
    transition_weights = transition_weights / transition_weights.mean().clamp_min(1e-6)

    weights = importance_weights.to(q_values.dtype) * transition_weights
    td1 = (weights * td1_per_item).mean()
    n_step = (weights * n_step_per_item).mean()
    margin = (weights * margin_per_item).mean()

    idle_no_op = q_values.new_zeros(())
    if self_actions is not None:
        alternative_q = q_values.clone()
        alternative_q[:, no_op_action_id] = torch.finfo(q_values.dtype).min
        idle_no_op_per_item = functional.relu(
            q_values[:, no_op_action_id]
            - alternative_q.max(dim=1).values
            + float(config.get("idle_no_op_advantage_margin", 0.0))
        )
        idle_weights = weights * idle_mask.to(weights.dtype)
        idle_no_op = (idle_weights * idle_no_op_per_item).sum() / idle_weights.sum().clamp_min(1e-6)
    l2 = parameter_l2([parameter for parameter in online_network.parameters() if parameter.requires_grad])
    total = (
        float(config["lambda_td1"]) * td1
        + float(config["lambda_nstep"]) * n_step
        + float(config["lambda_demo"]) * margin
        + float(config.get("lambda_idle_no_op", 0.0)) * idle_no_op
        + float(config["lambda_l2"]) * l2
    )
    return DQfDLossOutput(
        total,
        td1,
        n_step,
        margin,
        idle_no_op,
        l2,
        td_errors.detach(),
    )
