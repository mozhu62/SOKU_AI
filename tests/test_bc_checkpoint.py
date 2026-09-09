import copy
import tempfile
import unittest
from pathlib import Path

import torch

from soku_bc import checkpoint
from soku_bc.config import DEFAULTS
from soku_bc.learner import Learner
from tests.fixtures import tensor_observation, normalization


class BCCheckpointTests(unittest.TestCase):
    def test_bc_round_trip_model_optimizer_and_rng(self):
        config = copy.deepcopy(DEFAULTS)
        config['training'].update(device='cpu', amp=False, burn_in=0)
        learner = Learner(config)
        batch = {'observation': tensor_observation(1, 2), 'burn_lengths': torch.zeros(1, dtype=torch.long),
                 'joint_action_id': torch.full((1, 2), 192, dtype=torch.long),
                 'mask': torch.ones(1, 2, dtype=torch.bool)}
        learner.train_batch(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            norm = {**normalization(), 'split_hash': 'fixture-split'}
            checkpoint.save(path, learner, config, {'sha256': 'fixture-split'}, norm, 1, 1, 2, None, 0)
            expected_random = torch.rand(4)
            package = checkpoint.load(path)
            self.assertEqual(package['algorithm'], 'bc')
            self.assertEqual(package['spec']['output_semantics'], 'categorical_logits')
            self.assertNotIn('target', package)
            self.assertNotIn('online', package)
            restored = Learner(config)
            checkpoint.restore(package, restored, {'sha256': 'fixture-split'})
            self.assertTrue(torch.equal(torch.rand(4), expected_random))
            for key, value in learner.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[key]), key)
            original = learner.optimizer.state_dict()['state']
            recovered = restored.optimizer.state_dict()['state']
            self.assertEqual(original.keys(), recovered.keys())
            for key in original:
                for name, value in original[key].items():
                    self.assertTrue(torch.equal(value, recovered[key][name]))
            with self.assertRaisesRegex(ValueError, '固定数据划分'):
                checkpoint.restore(package, restored, {'sha256': 'different-split'})

    def test_cql_checkpoint_is_rejected_not_partially_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cql.pt'
            torch.save({'network_version': 'soku_cql_recurrent_joint432_v1', 'online': {}}, path)
            with self.assertRaisesRegex(ValueError, '仅接受 BC'):
                checkpoint.load(path)


if __name__ == '__main__':
    unittest.main()
