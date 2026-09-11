from __future__ import annotations

from typing import Any

import torch

from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.data.transition_builder import CATEGORICAL_PADDING_VALUE

from .interface import Observation


def empty_observation(history_len: int, max_objects: int) -> dict[str, torch.Tensor]:
    return {
        "state_continuous": torch.zeros(1, OBSERVATION_SHAPE.state_continuous),
        "state_categorical": torch.zeros(1, OBSERVATION_SHAPE.state_categorical, dtype=torch.long),
        "history_numerical": torch.zeros(
            1, history_len, OBSERVATION_SHAPE.history_numerical
        ),
        "history_categorical": torch.full(
            (1, history_len, OBSERVATION_SHAPE.history_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
        ),
        "history_mask": torch.zeros(1, history_len, dtype=torch.bool),
        "self_object_numerical": torch.zeros(
            1, max_objects, OBSERVATION_SHAPE.object_numerical
        ),
        "self_object_categorical": torch.full(
            (1, max_objects, OBSERVATION_SHAPE.object_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
        ),
        "self_object_mask": torch.zeros(1, max_objects, dtype=torch.bool),
        "opponent_object_numerical": torch.zeros(
            1, max_objects, OBSERVATION_SHAPE.object_numerical
        ),
        "opponent_object_categorical": torch.full(
            (1, max_objects, OBSERVATION_SHAPE.object_categorical),
            CATEGORICAL_PADDING_VALUE,
            dtype=torch.long,
        ),
        "opponent_object_mask": torch.zeros(1, max_objects, dtype=torch.bool),
    }


class MockSokuEnv:
    """只验证在线 Trainer 接口，不代表真实 TH123 动力学。"""

    def __init__(self, history_len: int = 32, max_objects: int = 32, horizon: int = 32) -> None:
        self.history_len = int(history_len)
        self.max_objects = int(max_objects)
        self.horizon = int(horizon)
        self.step_count = 0

    def reset(self) -> Observation:
        self.step_count = 0
        return empty_observation(self.history_len, self.max_objects)

    def step(self, action_id: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        observation = empty_observation(self.history_len, self.max_objects)
        observation["state_continuous"][0, 0] = float(self.step_count)
        terminated = self.step_count >= self.horizon
        return observation, 0.0, terminated, False, {
            "mock": True,
            "step": self.step_count,
            "action_id": int(action_id),
        }

    def close(self) -> None:
        self.step_count = 0

