from __future__ import annotations

from soku_ai.data.schemas import OBSERVATION_SHAPE
from soku_ai.data.resources_schema import SHAPE, MODEL_TYPE

from .q_network import SokuDuelingQNetwork


def build_model(config: dict) -> SokuDuelingQNetwork:
    model_config = config["model"]
    if model_config["model_type"] not in ("tcn_entity_dueling_dqn", MODEL_TYPE):
        raise ValueError(f"不支持的模型类型: {model_config['model_type']}")
    shape = SHAPE if model_config["model_type"] == MODEL_TYPE else OBSERVATION_SHAPE
    return SokuDuelingQNetwork(
        model_config,
        state_continuous_dim=shape.state_continuous,
        history_numerical_dim=shape.history_numerical,
        object_numerical_dim=shape.object_numerical,
    )
