from __future__ import annotations

import torch

from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.models.embeddings import GameEmbeddings
from soku_ai.models.tcn import TemporalEncoder


def test_tcn_batch_shape(model_config) -> None:
    embeddings = GameEmbeddings(model_config["model"])
    encoder = TemporalEncoder(
        OBSERVATION_SHAPE.history_numerical,
        embeddings,
        model_config["model"],
    )
    numerical = torch.randn(8, 32, OBSERVATION_SHAPE.history_numerical)
    categorical = torch.zeros(8, 32, OBSERVATION_SHAPE.history_categorical, dtype=torch.long)
    mask = torch.ones(8, 32, dtype=torch.bool)
    assert encoder(numerical, categorical, mask).shape == (8, 128)

