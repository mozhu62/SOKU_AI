import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soku_bc.action_space import encode, decode, previous_actions, START_ACTION_ID, ACTION_COUNT, to_controller, from_controller
from soku_bc.config import DEFAULTS
from soku_bc.dataset import read_shard, ReplayStore, split_replays
from tests.fixtures import raw_shard


class BCDatasetTests(unittest.TestCase):
    def test_all_144_actions_round_trip(self):
        ids = np.arange(144)
        np.testing.assert_array_equal(encode(*decode(ids)), ids)
        direction, buttons = to_controller(ids)
        self.assertEqual(buttons.shape, (ACTION_COUNT, 4))
        np.testing.assert_array_equal(from_controller(direction, buttons), ids)
        for invalid in (-1, ACTION_COUNT):
            with self.assertRaises(ValueError):
                decode(invalid)

    def test_previous_action_shift_and_boundaries(self):
        actions = np.arange(6)
        valid = np.array([True, True, False, True, True, False])
        terminal = np.array([False, True, False, False, False, False])
        previous, duration = previous_actions(actions, np.arange(1, 7), np.zeros(6), valid, terminal)
        np.testing.assert_array_equal(previous, [START_ACTION_ID, 0, START_ACTION_ID, START_ACTION_ID, 3, 4])
        self.assertAlmostEqual(float(duration[4, 0]), 4/60)

    def test_rewards_not_loaded_and_source_is_not_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'demo.npz'
            raw = raw_shard()
            raw['rewards'][:] = np.nan
            raw['action_buttons'][1, 4:] = 1
            np.savez(path, **raw)
            before = path.read_bytes()
            shard = read_shard(path, True)
            self.assertNotIn('rewards', shard)
            self.assertNotIn('terminated', shard)
            self.assertEqual(shard['joint_action_id'][1], encode(5, 0))
            self.assertEqual(shard['previous_joint_action_id'][2], encode(5, 0))
            self.assertEqual(shard['card_input_ignored_rows'], 1)
            self.assertEqual(len(shard['joint_action_id']), len(raw['episode_id']))
            self.assertEqual(path.read_bytes(), before)
            raw.pop('rewards')
            np.savez(path, **raw)
            read_shard(path, True)

    def test_fixed_split_training_normalization_and_bc_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(5):
                raw = raw_shard(count=8)
                raw['state_continuous'][:] = index
                raw['action_horizontal'][:] = index % 3 - 1
                np.savez(root / f'{index}.npz', **raw)
            config = copy.deepcopy(DEFAULTS)
            config['data'].update(directory=str(root), split_file=str(root/'split.json'))
            split = split_replays(config, lambda _: None)
            self.assertEqual((len(split['train']), len(split['validation'])), (4, 1))
            self.assertEqual(split, split_replays(config, lambda _: None))
            store = ReplayStore(config, split, lambda _: None)
            expected = np.mean([int(Path(name).stem) for name in split['train']])
            np.testing.assert_allclose(store.normalization['state']['mean'], expected)
            cfg = {**config['training'], 'burn_in': 3, 'sequence_length': 10, 'batch_size': 4}
            batch = store.sample(np.random.default_rng(0), cfg)
            self.assertEqual(set(batch), {'observation', 'burn_lengths', 'joint_action_id', 'mask'})
            self.assertEqual(batch['joint_action_id'].shape, (4, 10))
            self.assertEqual(batch['observation']['state_continuous'].shape, (4, 13, 18))
            self.assertTrue(batch['mask'].any())
            self.assertTrue((~batch['mask']).any())
            self.assertEqual(batch['observation']['self_object_numerical'].shape[-2], 3)
            self.assertEqual(batch['observation']['opponent_object_numerical'].shape[-2], 3)
            self.assertNotIn('joint_action_id', batch['observation'])


if __name__ == '__main__':
    unittest.main()
