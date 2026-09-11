from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import torch


Observation: TypeAlias = Mapping[str, torch.Tensor]


@runtime_checkable
class SokuEnv(Protocol):
    def reset(self) -> Observation:
        ...

    def step(self, action_id: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class OpponentPolicy(Protocol):
    def select_action(self, observation: Observation) -> int:
        ...

