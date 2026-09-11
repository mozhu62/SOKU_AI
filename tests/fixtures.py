import json

import numpy as np
import torch

from soku_bc.schema import manifest
from soku_bc.action_space import ACTION_SCHEMA


def raw_shard(count=6):
    metadata = {**manifest(), "action_shift": 1, "continuous_normalized": False}
    result = {"metadata_json": np.asarray(json.dumps(metadata)),
              "episode_id": np.zeros(count, np.int32),
              "state_continuous": np.zeros((count, 18), np.float32),
              "state_categorical": np.zeros((count, 5), np.int64),
              "tactical_state": np.zeros((count, 9), np.float32),
              "action_horizontal": np.zeros(count, np.int8), "action_vertical": np.zeros(count, np.int8),
              "action_buttons": np.zeros((count, 6), np.uint8), "action_duration": np.arange(1, count + 1),
              "transition_valid": np.arange(count) < count - 1, "rewards": np.zeros(count, np.float32),
              "terminated": np.zeros(count, bool)}
    for side in ("self", "opponent"):
        result.update({f"{side}_object_numerical": np.zeros((0, 8), np.float32),
                       f"{side}_object_categorical": np.zeros((0, 2), np.int64),
                       f"{side}_object_offsets": np.zeros(count + 1, np.int64),
                       f"{side}_skill_valid_mask": np.full(count, 15, np.uint8),
                       f"{side}_skill_variants": np.zeros((count, 4), np.int8),
                       f"{side}_skill_levels": np.zeros((count, 4), np.int8),
                       f"{side}_skill_effective_levels": np.zeros((count, 4), np.int8),
                       f"{side}_card_state": np.tile([0, 0, 0, 65535, 0, 5, 0, 0], (count, 1)).astype(np.int32),
                       f"{side}_hand_card_ids": np.full((count, 16), 65535, np.uint16),
                       f"{side}_hand_card_costs": np.zeros((count, 16), np.uint16),
                       f"{side}_hand_mask": np.zeros((count, 16), np.uint8)})
    return result


def normalization():
    result = {key: {"mean": [0.0] * size, "std": [1.0] * size, "count": 1}
              for key, size in (("state", 18), ("objects", 8), ("optional_state", 4))}
    result["optional_state"]["counts"] = [0] * 4
    result.update(action_schema=ACTION_SCHEMA, action_shift=1)
    return result


def tensor_observation(batch=2, length=5):
    shape = (batch, length)
    result = {"state_continuous": torch.zeros(*shape, 18), "state_categorical": torch.zeros(*shape, 5, dtype=torch.long),
              "tactical_state": torch.zeros(*shape, 9), "state_optional_continuous": torch.zeros(*shape, 4),
              "state_optional_mask": torch.zeros(*shape, 4, dtype=torch.bool),
              "previous_joint_action_id": torch.full(shape, 144, dtype=torch.long),
              "previous_action_duration": torch.zeros(*shape, 1)}
    for side in ("self", "opponent"):
        result.update({f"{side}_object_numerical": torch.zeros(*shape, 3, 8),
                       f"{side}_object_categorical": torch.zeros(*shape, 3, 2, dtype=torch.long),
                       f"{side}_object_mask": torch.zeros(*shape, 3, dtype=torch.bool),
                       f"{side}_skill_categorical": torch.zeros(*shape, 4, dtype=torch.long),
                       f"{side}_skill_mask": torch.zeros(*shape, 4, dtype=torch.bool)})
    return result
