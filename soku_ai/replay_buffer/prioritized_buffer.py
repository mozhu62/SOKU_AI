from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


class SumTree:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("SumTree capacity 必须大于 0")
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.capacity = int(capacity)
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(tree_capacity * 2, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, index: int) -> float:
        return float(self.tree[self.tree_capacity + int(index)])

    def update(self, index: int, value: float) -> None:
        self.update_many([index], [value])

    def update_many(self, indices: Sequence[int], values: Sequence[float]) -> None:
        index_array = np.asarray(indices, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float64)
        if index_array.shape != value_array.shape:
            raise ValueError("SumTree 批量更新的索引和值长度不一致")
        if index_array.size == 0:
            return

        # 同一批可能重复采到相同叶子，保持原实现的“最后一次更新生效”语义。
        reversed_indices = index_array[::-1]
        _, reversed_positions = np.unique(reversed_indices, return_index=True)
        keep = index_array.size - 1 - reversed_positions
        index_array = index_array[keep]
        value_array = value_array[keep]
        nodes = self.tree_capacity + index_array
        self.tree[nodes] = value_array
        nodes = np.unique(nodes // 2)
        while nodes.size and nodes[0] >= 1:
            self.tree[nodes] = self.tree[nodes * 2] + self.tree[nodes * 2 + 1]
            nodes = np.unique(nodes // 2)
            nodes = nodes[nodes >= 1]

    def build(self, values: Sequence[float]) -> None:
        """一次性构造所有叶子，避免初始化时逐样本回溯整棵树。"""
        value_array = np.asarray(values, dtype=np.float64)
        if value_array.shape != (self.capacity,):
            raise ValueError("SumTree 初始化值数量与 capacity 不一致")
        self.tree.fill(0.0)
        start = self.tree_capacity
        self.tree[start : start + self.capacity] = value_array
        for node in range(self.tree_capacity - 1, 0, -1):
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1]

    def find_prefix(self, mass: float) -> int:
        return int(self.find_prefix_many([mass])[0])

    def find_prefix_many(self, masses: Sequence[float]) -> np.ndarray:
        if self.total <= 0:
            raise RuntimeError("不能从空 SumTree 采样")
        remaining = np.asarray(masses, dtype=np.float64).copy()
        remaining = np.clip(remaining, 0.0, np.nextafter(self.total, 0.0))
        nodes = np.ones(remaining.shape, dtype=np.int64)
        while int(nodes.max(initial=1)) < self.tree_capacity:
            left = nodes * 2
            go_right = remaining >= self.tree[left]
            remaining = np.where(go_right, remaining - self.tree[left], remaining)
            nodes = left + go_right.astype(np.int64)
        return np.minimum(nodes - self.tree_capacity, self.capacity - 1)


@dataclass(frozen=True)
class PrioritizedSample:
    indices: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray


class PrioritizedIndexBuffer:
    """为磁盘 Dataset 维护优先级，只存索引而不复制高维 Observation。"""

    def __init__(
        self,
        size: int,
        *,
        alpha: float,
        priority_epsilon: float,
        demo_priority_bonus: float = 0.0,
        is_demo: np.ndarray | None = None,
    ) -> None:
        self.size = int(size)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.demo_priority_bonus = float(demo_priority_bonus)
        self.is_demo = (
            np.ones(self.size, dtype=np.bool_)
            if is_demo is None
            else np.asarray(is_demo, dtype=np.bool_)
        )
        if len(self.is_demo) != self.size:
            raise ValueError("is_demo 长度与索引数量不一致")
        self.tree = SumTree(self.size)
        initial = np.ones(self.size, dtype=np.float64)
        self.tree.build(
            self._scaled_priority(initial, np.arange(self.size, dtype=np.int64))
        )

    def _scaled_priority(self, priorities: np.ndarray, indices: np.ndarray) -> np.ndarray:
        bonus = self.is_demo[indices].astype(np.float64) * self.demo_priority_bonus
        return np.power(np.abs(priorities) + bonus + self.priority_epsilon, self.alpha)

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        index_array = np.asarray(indices, dtype=np.int64)
        priority_array = np.asarray(priorities, dtype=np.float64)
        if np.any(index_array < 0) or np.any(index_array >= self.size):
            raise IndexError("PER priority index 越界")
        if not np.all(np.isfinite(priority_array)):
            raise ValueError("PER priority 包含 NaN 或 Inf")
        scaled = self._scaled_priority(priority_array, index_array)
        self.tree.update_many(index_array, scaled)

    def sample(
        self,
        batch_size: int,
        beta: float,
        rng: np.random.Generator,
    ) -> PrioritizedSample:
        total = self.tree.total
        if total <= 0:
            raise RuntimeError("PER 中没有可采样数据")
        segment = total / int(batch_size)
        masses = (
            np.arange(batch_size, dtype=np.float64) * segment
            + rng.random(batch_size) * segment
        )
        indices = self.tree.find_prefix_many(masses)
        probabilities = self.tree.tree[self.tree.tree_capacity + indices] / total
        weights = np.power(self.size * probabilities, -float(beta))
        weights /= max(float(weights.max()), 1e-12)
        return PrioritizedSample(indices, weights.astype(np.float32), probabilities.astype(np.float32))

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "alpha": self.alpha,
            "priority_epsilon": self.priority_epsilon,
            "demo_priority_bonus": self.demo_priority_bonus,
            "tree": self.tree.tree.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("Checkpoint PER 大小与当前 Dataset 不一致")
        tree = np.asarray(state["tree"], dtype=np.float64)
        if tree.shape != self.tree.tree.shape:
            raise ValueError("Checkpoint PER tree shape 不一致")
        self.tree.tree[...] = tree


class PrioritizedReplayBuffer:
    """在线阶段用的环形存储；Demo 占据固定前缀，永远不会被 agent 数据覆盖。"""

    def __init__(
        self,
        capacity: int,
        demonstrations: Iterable[Any] = (),
        *,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        demo_priority_bonus: float = 1.0,
    ) -> None:
        self.capacity = int(capacity)
        self.storage: list[Any | None] = [None] * self.capacity
        demos = list(demonstrations)
        if len(demos) >= self.capacity:
            raise ValueError("Replay Buffer 必须为未来 agent 数据保留至少一个槽位")
        self.demo_count = len(demos)
        self.size = self.demo_count
        self.agent_position = self.demo_count
        self.index_buffer = PrioritizedIndexBuffer(
            self.capacity,
            alpha=alpha,
            priority_epsilon=priority_epsilon,
            demo_priority_bonus=demo_priority_bonus,
            is_demo=np.arange(self.capacity) < self.demo_count,
        )
        for index, transition in enumerate(demos):
            self.storage[index] = transition
        # 尚未写入的 agent 槽位不能参与采样
        for index in range(self.demo_count, self.capacity):
            self.index_buffer.tree.update(index, 0.0)

    def add(self, transition: Any, priority: float | None = None) -> int:
        index = self.agent_position
        self.storage[index] = transition
        self.size = min(self.capacity, self.size + 1)
        self.agent_position += 1
        if self.agent_position >= self.capacity:
            self.agent_position = self.demo_count
        effective_priority = 1.0 if priority is None else float(priority)
        self.index_buffer.update_priorities([index], [effective_priority])
        return index

    def sample(
        self,
        batch_size: int,
        beta: float,
        rng: np.random.Generator,
    ) -> tuple[list[Any], PrioritizedSample]:
        if self.size < batch_size:
            raise RuntimeError("Replay Buffer 数据量小于 batch_size")
        sample = self.index_buffer.sample(batch_size, beta, rng)
        transitions = [self.storage[int(index)] for index in sample.indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("PER 采样到了尚未初始化的槽位")
        return transitions, sample

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        self.index_buffer.update_priorities(indices, priorities)


def linear_beta(
    step: int,
    *,
    beta_start: float,
    beta_end: float,
    anneal_steps: int,
) -> float:
    progress = min(max(step, 0) / max(int(anneal_steps), 1), 1.0)
    return float(beta_start + (beta_end - beta_start) * progress)
