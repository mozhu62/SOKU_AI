"""Temporal + Entity + Dueling Q 网络。"""

from .factory import build_model
from .q_network import SokuDuelingQNetwork

__all__ = ["SokuDuelingQNetwork", "build_model"]

