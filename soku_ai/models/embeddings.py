from __future__ import annotations

import torch
from torch import nn

from soku_ai.data.transition_builder import CATEGORICAL_PADDING_VALUE
from soku_ai.data.resources_schema import MODEL_TYPE


class SafeEmbedding(nn.Module):
    """将越界 ID 映射到 unknown，并为历史/对象 padding 保留独立索引。"""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.unknown_index = self.vocab_size
        self.padding_index = self.vocab_size + 1
        self.embedding = nn.Embedding(
            self.vocab_size + 2,
            embedding_dim,
            padding_idx=self.padding_index,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.long()
        padding = ids == CATEGORICAL_PADDING_VALUE
        unknown = (ids < 0) | (ids >= self.vocab_size)
        mapped = torch.where(unknown, self.unknown_index, ids)
        mapped = torch.where(padding, self.padding_index, mapped)
        return self.embedding(mapped)


class GameEmbeddings(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.action = SafeEmbedding(config["action_vocab_size"], config["action_embedding_dim"])
        self.block = SafeEmbedding(config["block_vocab_size"], config["block_embedding_dim"])
        if config.get("model_type") != MODEL_TYPE:
            self.character = SafeEmbedding(config["character_vocab_size"], config["character_embedding_dim"])
        self.weather = SafeEmbedding(config["weather_vocab_size"], config["weather_embedding_dim"])
        if config.get("model_type") != MODEL_TYPE:
            self.stage = SafeEmbedding(config["stage_vocab_size"], config["stage_embedding_dim"])
