from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from soku_ai.env.interface import SokuEnv
from soku_ai.replay_buffer.prioritized_buffer import PrioritizedReplayBuffer


@dataclass
class OnlineTrainerState:
    environment_steps: int = 0
    optimizer_steps: int = 0


class OnlineDQNTrainer:
    """只依赖 SokuEnv 的在线训练骨架，不包含任何 TH123 进程实现。"""

    def __init__(
        self,
        env: SokuEnv,
        model: torch.nn.Module,
        replay_buffer: PrioritizedReplayBuffer,
        *,
        num_actions: int,
        seed: int,
    ) -> None:
        self.env = env
        self.model = model
        self.replay_buffer = replay_buffer
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)
        self.state = OnlineTrainerState()
        self.observation: Any | None = None

    def select_action(self, observation: Any, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_actions))
        with torch.no_grad():
            q_values = self.model(observation)
        return int(q_values.argmax(dim=-1).item())

    def collect_step(self, epsilon: float) -> dict[str, Any]:
        if self.observation is None:
            self.observation = self.env.reset()
        action = self.select_action(self.observation, epsilon)
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        transition = {
            "observation": self.observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": terminated or truncated,
            "is_demo": False,
        }
        self.replay_buffer.add(transition)
        self.state.environment_steps += 1
        self.observation = None if terminated or truncated else next_observation
        return info

    def close(self) -> None:
        self.env.close()
