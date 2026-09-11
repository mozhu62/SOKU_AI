from __future__ import annotations

import torch
from torch import nn

from .embeddings import GameEmbeddings
from soku_ai.data.resources_schema import MODEL_TYPE


class StateEncoder(nn.Module):
    def __init__(
        self,
        continuous_dim: int,
        embeddings: GameEmbeddings,
        model_config: dict,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.compact = model_config.get("model_type") == MODEL_TYPE
        embedded_dim = (
            model_config["action_embedding_dim"] * 2
            + model_config["block_embedding_dim"] * 2
            + model_config["weather_embedding_dim"]
        )
        if not self.compact:
            embedded_dim += model_config["character_embedding_dim"] + model_config["weather_embedding_dim"] + model_config["stage_embedding_dim"]
        hidden = int(model_config["state_hidden_dim"])
        self.network = nn.Sequential(
            nn.Linear(continuous_dim + embedded_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.output_dim = hidden

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if self.compact:
            # 原数据没有对手角色、显示天气、场景编号，不再建立这些输入或 embedding。
            embedded = torch.cat([self.embeddings.action(categorical[:, 0]), self.embeddings.block(categorical[:, 1]),
                                  self.embeddings.action(categorical[:, 2]), self.embeddings.block(categorical[:, 3]),
                                  self.embeddings.weather(categorical[:, 4])], dim=-1)
            return self.network(torch.cat([continuous, embedded], dim=-1))
        embedded = torch.cat(
            [
                self.embeddings.action(categorical[:, 0]),
                self.embeddings.block(categorical[:, 1]),
                self.embeddings.character(categorical[:, 2]),
                self.embeddings.action(categorical[:, 3]),
                self.embeddings.block(categorical[:, 4]),
                self.embeddings.weather(categorical[:, 5]),
                self.embeddings.weather(categorical[:, 6]),
                self.embeddings.stage(categorical[:, 7]),
            ],
            dim=-1,
        )
        return self.network(torch.cat([continuous, embedded], dim=-1))
