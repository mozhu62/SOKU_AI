import copy
import tempfile
import unittest
from pathlib import Path

import torch

from soku_bc import checkpoint
from soku_bc.config import DEFAULTS, NETWORK_VERSION, validate
from soku_bc.experiments import comparison_conditions, experiment_path
from soku_bc.learner import Learner
from soku_bc.models import BCNetwork
from soku_bc.temporal import TemporalConvEncoder
from tests.fixtures import normalization, tensor_observation


def tcn_config():
    config = copy.deepcopy(DEFAULTS)
    config['training'].update(device='cpu', amp=False, burn_in=31, sequence_length=4, batch_size=1)
    return config


class TemporalExperimentTests(unittest.TestCase):
    """仅交付验收源码；按项目要求，本次不自动执行这些测试。"""

    def test_tcn_has_two_convolutions_per_block_and_exact_32_frame_receptive_field(self):
        encoder = TemporalConvEncoder(228, 256, 256).eval()
        self.assertTrue(all(hasattr(block, 'conv1') and hasattr(block, 'conv2') for block in encoder.blocks))
        features = torch.randn(1, 70, 228, requires_grad=True)
        result = encoder(features)
        result[:, 50].square().sum().backward()
        self.assertEqual(features.grad[:, :19].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 51:].abs().sum().item(), 0)
        self.assertGreater(features.grad[:, 19].abs().sum().item(), 0)

    def test_tcn_batch_and_stream_match_including_short_prefix(self):
        model = BCNetwork(tcn_config()['model']).eval()
        obs = tensor_observation(1, 35)
        obs['state_continuous'] = torch.randn(1, 35, 18)
        with torch.no_grad():
            expected = model(obs, 31, torch.tensor([3]))
            memory = None
            for index in range(3):
                _, memory = model.step_logits({key: value[:, index] for key, value in obs.items()}, memory)
            outputs = []
            for index in range(31, 35):
                value, memory = model.step_logits({key: array[:, index] for key, array in obs.items()}, memory)
                outputs.append(value)
            self.assertTrue(torch.allclose(expected, torch.stack(outputs, 1), atol=1e-5))
            full = model(obs)
            memory, outputs = None, []
            for index in range(35):
                value, memory = model.step_logits({key: array[:, index] for key, array in obs.items()}, memory)
                outputs.append(value)
                self.assertLessEqual(memory.shape[1], 31)
            self.assertTrue(torch.allclose(full, torch.stack(outputs, 1), atol=1e-5))

    def test_network_contains_no_gru_and_all_active_modules_can_update(self):
        learner = Learner(tcn_config())
        self.assertFalse(hasattr(learner.model, 'gru'))
        self.assertFalse(hasattr(learner.model, 'memory_fusion'))
        batch = {
            'observation': tensor_observation(1, 35),
            'burn_lengths': torch.tensor([31]),
            'joint_action_id': torch.full((1, 4), 64, dtype=torch.long),
            'mask': torch.ones(1, 4, dtype=torch.bool),
        }
        result = learner.train_batch(batch, diagnostics=True)
        self.assertGreater(result['module_changes']['tcn'], 0)
        self.assertGreater(result['module_changes']['current_encoder'], 0)
        self.assertTrue(all(not row['bypassed'] for row in learner.model.module_status().values()))

    def test_new_checkpoint_is_strict_and_old_architecture_is_rejected(self):
        config = tcn_config()
        split = {'sha256': 'fixture'}
        norm = {**normalization(), 'split_hash': 'fixture'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            learner = Learner(config)
            checkpoint.save(path, learner, config, split, norm, 0, 0, 0, None, 0)
            package = checkpoint.load(path)
            self.assertEqual(package['network_version'], NETWORK_VERSION)
            checkpoint.restore(package, Learner(config), split)
            old = torch.load(path, weights_only=True)
            old['network_version'] = 'soku_bc_wide_tcn32_joint432_v2'
            old['spec']['network_version'] = 'soku_bc_wide_tcn32_joint432_v2'
            torch.save(old, path)
            with self.assertRaisesRegex(ValueError, '旧 Joint432 权重不允许部分加载'):
                checkpoint.load(path)

    def test_fixed_mode_validation_and_comparison_conditions(self):
        config = tcn_config()
        validate(config)
        config['training']['burn_in'] = 16
        with self.assertRaisesRegex(ValueError, '31'):
            validate(config)
        config['training']['burn_in'] = 31
        changed = copy.deepcopy(config)
        changed['training']['frozen_modules'] = ['tcn']
        self.assertNotEqual(
            comparison_conditions(config, 'split', normalization())[1],
            comparison_conditions(changed, 'split', normalization())[1],
        )
        invalid = copy.deepcopy(config)
        invalid['model']['temporal_mode'] = 'gru'
        with self.assertRaisesRegex(ValueError, '删除 GRU'):
            validate(invalid)
        with self.assertRaises(ValueError):
            experiment_path('../unsafe')


if __name__ == '__main__':
    unittest.main()
