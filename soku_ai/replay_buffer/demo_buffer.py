from __future__ import annotations

from .prioritized_buffer import PrioritizedIndexBuffer


class DemonstrationIndexBuffer(PrioritizedIndexBuffer):
    """语义化别名：其所有索引均为永久保留的专家 Transition。"""

