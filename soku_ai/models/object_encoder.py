from __future__ import annotations

import torch
from torch import nn

from .embeddings import GameEmbeddings


class ObjectEncoder(nn.Module):
    """双方对象共享参数，输出各自的 masked mean + max 特征。"""

    def __init__(
        self,
        numerical_dim: int,
        embeddings: GameEmbeddings,
        model_config: dict,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        input_dim = (
            numerical_dim
            + model_config["action_embedding_dim"]
            + model_config["block_embedding_dim"]
        )
        entity_dim = int(model_config["object_entity_dim"])
        output_dim = int(model_config["object_output_dim"])
        self.entity_network = nn.Sequential(
            nn.Linear(input_dim, entity_dim),
            nn.SiLU(),
            nn.LayerNorm(entity_dim),
            nn.Dropout(float(model_config["object_dropout"])),
            nn.Linear(entity_dim, entity_dim),
            nn.SiLU(),
        )
        self.output_projection = (
            nn.Identity()
            if output_dim == entity_dim * 2
            else nn.Linear(entity_dim * 2, output_dim)
        )
        self.output_dim = output_dim

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = torch.cat(
            [
                self.embeddings.action(categorical[:, :, 0]),
                self.embeddings.block(categorical[:, :, 1]),
            ],
            dim=-1,
        )
        entity = self.entity_network(torch.cat([numerical, embedded], dim=-1))
        float_mask = mask.unsqueeze(-1).to(entity.dtype)
        entity = entity * float_mask
        count = float_mask.sum(dim=1).clamp_min(1.0)
        mean_pool = entity.sum(dim=1) / count
        masked = entity.masked_fill(~mask.unsqueeze(-1), torch.finfo(entity.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return self.output_projection(torch.cat([mean_pool, max_pool], dim=-1))

