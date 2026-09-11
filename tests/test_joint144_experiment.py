import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from soku_bc.action_space import ACTION_COUNT, START_ACTION_ID, encode, from_controller
from soku_bc.config import DEFAULTS
from soku_bc.dataset import read_shard
from soku_bc.models import BCNetwork
from soku_bc.resources import IGNORED_RESOURCE_SUFFIXES, resource_observation
from tests.fixtures import raw_shard, tensor_observation


class Joint144ExperimentTests(unittest.TestCase):
    """本次只交付验收用例源码，不自动运行模型或测试。"""

    def test_card_only_changes_become_hold_without_removing_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.npz'
            raw = raw_shard(8)
            raw['action_buttons'][:4, 4:] = [[0, 0], [1, 0], [0, 1], [1, 1]]
            np.savez(path, **raw)
            before = path.read_bytes()
            shard = read_shard(path, True)
            np.testing.assert_array_equal(shard['joint_action_id'], np.full(8, encode(5, 0)))
            self.assertEqual(shard['previous_joint_action_id'][0], START_ACTION_ID)
            np.testing.assert_array_equal(shard['previous_joint_action_id'][1:7], np.full(6, encode(5, 0)))
            self.assertEqual(int(shard['card_input_ignored_rows']), 3)
            self.assertEqual(path.read_bytes(), before)

    def test_removed_resources_are_not_required_or_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'resources.npz'
            raw = raw_shard()
            np.savez(path, **raw)
            full = read_shard(path, True)
            for side in ('self', 'opponent'):
                for suffix in IGNORED_RESOURCE_SUFFIXES:
                    self.assertNotIn(f'{side}_{suffix}', full)
                    raw.pop(f'{side}_{suffix}')
            np.savez(path, **raw)
            reduced = read_shard(path, True)
            first = resource_observation(full, np.arange(3))
            second = resource_observation(reduced, np.arange(3))
            self.assertEqual(set(first), {'self_skill_categorical', 'self_skill_mask',
                                          'opponent_skill_categorical', 'opponent_skill_mask'})
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
                self.assertEqual(first[key].shape, (3, 4))

    def test_all_legacy_actions_project_to_the_same_direction_and_combat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'all_actions.npz'
            old_ids = np.arange(432, dtype=np.int64)
            new_ids, card = old_ids // 3, old_ids % 3
            direction, combat = new_ids // 16 + 1, new_ids % 16
            raw = raw_shard(len(old_ids))
            raw['action_horizontal'] = ((direction - 1) % 3 - 1).astype(np.int8)
            raw['action_vertical'] = (1 - (direction - 1) // 3).astype(np.int8)
            raw['action_buttons'] = np.stack([*((combat >> bit) & 1 for bit in range(4)),
                                              card == 1, card == 2], -1).astype(np.uint8)
            raw['joint_action_id'] = old_ids
            np.savez(path, **raw)
            shard = read_shard(path, True)
            np.testing.assert_array_equal(shard['joint_action_id'], new_ids)

    def test_model_has_no_card_or_level_parameters(self):
        model = BCNetwork(copy.deepcopy(DEFAULTS['model'])).eval()
        self.assertEqual(model.spec['current'], {'input_dim': 228, 'output_dim': 256, 'layers': 1})
        self.assertEqual(model.spec['fusion']['input_dim'], 768)
        self.assertEqual(model.policy_head.out_features, ACTION_COUNT)
        self.assertFalse(any('card' in name or 'level' in name for name, _ in model.named_parameters()))
        observation = tensor_observation(1, 32)
        with torch.no_grad():
            self.assertEqual(model.state_features(observation).shape, (1, 32, 228))
            self.assertEqual(model(observation).shape, (1, 32, 144))
        with self.assertRaises(ValueError):
            from_controller(5, [0, 0, 0, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
