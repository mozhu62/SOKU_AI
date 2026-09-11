from __future__ import annotations

import torch

from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.models.embeddings import GameEmbeddings
from soku_ai.models.object_encoder import ObjectEncoder


def test_object_counts_and_empty_are_finite(model_config) -> None:
    embeddings = GameEmbeddings(model_config["model"])
    encoder = ObjectEncoder(
        OBSERVATION_SHAPE.object_numerical,
        embeddings,
        model_config["model"],
    )
    for count in (0, 1, 8, 32):
        numerical = torch.randn(2, 32, OBSERVATION_SHAPE.object_numerical)
        categorical = torch.zeros(2, 32, OBSERVATION_SHAPE.object_categorical, dtype=torch.long)
        mask = torch.arange(32).unsqueeze(0).expand(2, -1) < count
        output = encoder(numerical, categorical, mask)
        assert output.shape == (2, model_config["model"]["object_output_dim"])
        assert torch.isfinite(output).all()

