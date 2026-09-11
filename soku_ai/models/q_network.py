from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .dueling_head import DuelingQHead
from .embeddings import GameEmbeddings
from .object_encoder import ObjectEncoder
from .state_encoder import StateEncoder
from .tcn import TemporalEncoder
from soku_ai.data.resources_schema import MODEL_TYPE, ARCHITECTURE_VERSION, OBSERVATION_VERSION
from soku_ai.data.schemas import SCHEMA_VERSION


class SokuDuelingQNetwork(nn.Module):
    architecture_version = "tcn_entity_dueling_dqn_v1"

    def __init__(
        self,
        model_config: dict,
        *,
        state_continuous_dim: int,
        history_numerical_dim: int,
        object_numerical_dim: int,
    ) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        self.observation_schema_version = SCHEMA_VERSION
        if model_config.get("model_type") == MODEL_TYPE:
            self.architecture_version = ARCHITECTURE_VERSION
            self.observation_schema_version = OBSERVATION_VERSION
        self.embeddings = GameEmbeddings(model_config)
        self.state_encoder = StateEncoder(
            state_continuous_dim, self.embeddings, model_config
        )
        self.temporal_encoder = TemporalEncoder(
            history_numerical_dim, self.embeddings, model_config
        )
        self.object_encoder = ObjectEncoder(
            object_numerical_dim, self.embeddings, model_config
        )
        fusion_input = (
            self.state_encoder.output_dim
            + self.temporal_encoder.output_dim
            + self.object_encoder.output_dim * 2
        )
        fused_dim = int(model_config["fused_dim"])
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, int(model_config["fusion_hidden_dim"])),
            nn.SiLU(),
            nn.LayerNorm(int(model_config["fusion_hidden_dim"])),
            nn.Linear(int(model_config["fusion_hidden_dim"]), fused_dim),
            nn.SiLU(),
        )
        self.head = DuelingQHead(
            fused_dim,
            int(model_config["dueling_hidden_dim"]),
            int(model_config["num_actions"]),
        )

    def encode(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        state = self.state_encoder(
            observation["state_continuous"], observation["state_categorical"]
        )
        temporal = self.temporal_encoder(
            observation["history_numerical"],
            observation["history_categorical"],
            observation["history_mask"],
        )
        self_objects = self.object_encoder(
            observation["self_object_numerical"],
            observation["self_object_categorical"],
            observation["self_object_mask"],
        )
        opponent_objects = self.object_encoder(
            observation["opponent_object_numerical"],
            observation["opponent_object_categorical"],
            observation["opponent_object_mask"],
        )
        return self.fusion(torch.cat([state, temporal, self_objects, opponent_objects], dim=-1))

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(observation))
