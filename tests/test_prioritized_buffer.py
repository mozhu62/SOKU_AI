from __future__ import annotations

import numpy as np

from soku_ai.replay_buffer.prioritized_buffer import (
    PrioritizedIndexBuffer,
    PrioritizedReplayBuffer,
)


def test_priority_update_changes_sampling() -> None:
    buffer = PrioritizedIndexBuffer(
        4,
        alpha=1.0,
        priority_epsilon=1e-6,
        demo_priority_bonus=0.0,
    )
    buffer.update_priorities([0, 1, 2, 3], [1.0, 1.0, 1.0, 100.0])
    rng = np.random.default_rng(1)
    samples = [int(buffer.sample(1, 0.4, rng).indices[0]) for _ in range(200)]
    assert samples.count(3) > 150


def test_agent_ring_never_overwrites_demo() -> None:
    buffer = PrioritizedReplayBuffer(5, demonstrations=["demo0", "demo1"])
    for index in range(10):
        buffer.add(f"agent{index}")
    assert buffer.storage[:2] == ["demo0", "demo1"]

