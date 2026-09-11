from __future__ import annotations

import torch

from soku_ai.rl.dqfd_loss import compute_dqfd_loss, large_margin_loss


def test_margin_loss_decreases_when_expert_q_is_high() -> None:
    actions = torch.tensor([1])
    demo = torch.tensor([True])
    low = large_margin_loss(torch.tensor([[1.0, 0.0, 2.0]]), actions, demo, 0.8)
    high = large_margin_loss(torch.tensor([[1.0, 5.0, 2.0]]), actions, demo, 0.8)
    assert high.item() < low.item()
    assert high.item() == 0.0


def test_full_dqfd_loss_backward() -> None:
    model = torch.nn.Linear(4, 3)
    values = model(torch.randn(2, 4))
    config = {
        "margin": 0.8,
        "lambda_td1": 1.0,
        "lambda_nstep": 1.0,
        "lambda_demo": 1.0,
        "lambda_l2": 1e-5,
    }
    loss = compute_dqfd_loss(
        q_values=values,
        actions=torch.tensor([0, 1]),
        td1_targets=torch.tensor([1.0, -1.0]),
        n_step_targets=torch.tensor([0.5, -0.5]),
        is_demo=torch.tensor([True, True]),
        importance_weights=torch.ones(2),
        online_network=model,
        config=config,
    )
    loss.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

