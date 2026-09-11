from __future__ import annotations

from collections.abc import Sequence


def calculate_n_step_return(
    rewards: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float,
    n_step: int,
) -> tuple[float, float, bool, int]:
    total = 0.0
    discount = 1.0
    terminal = False
    steps = 0
    for reward, done in zip(rewards[:n_step], dones[:n_step], strict=False):
        total += discount * float(reward)
        steps += 1
        terminal = bool(done)
        discount *= float(gamma)
        if terminal:
            break
    return total, discount, terminal, steps

