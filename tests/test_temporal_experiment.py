import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from soku_bc import checkpoint
from soku_bc.config import DEFAULTS, NETWORK_VERSION, TCN_NETWORK_VERSION, validate
from soku_bc.experiments import comparison_conditions, experiment_path
from soku_bc.learner import Learner
from soku_bc.models import BCNetwork
from soku_bc.temporal import TemporalConvEncoder
from tests.fixtures import normalization, tensor_observation


def tcn_config():
    config = copy.deepcopy(DEFAULTS)
    config['model']['temporal_mode'] = 'tcn'
    config['training'].update(device='cpu', amp=False, burn_in=31, sequence_length=4, batch_size=1)
    return config


class TemporalExperimentTests(unittest.TestCase):
    """仅交付验收源码；按项目要求，本次不自动执行这些测试。"""

    def test_tcn_has_exact_32_frame_causal_receptive_field(self):
        encoder = TemporalConvEncoder().eval()
        features = torch.randn(1, 70, 256, requires_grad=True)
        result = encoder(features)
        result[:, 50].square().sum().backward()
        self.assertEqual(features.grad[:, :19].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, 51:].abs().sum().item(), 0)
        self.assertGreater(features.grad[:, 19].abs().sum().item(), 0)
        changed = features.detach().clone()
        changed[:, 51:] = torch.randn_like(changed[:, 51:]) * 50
        with torch.no_grad():
            self.assertTrue(torch.allclose(result[:, :51], encoder(changed)[:, :51], atol=1e-5))

    def test_tcn_batch_and_stream_match_including_short_prefix(self):
        model = BCNetwork(tcn_config()['model']).eval()
        obs = tensor_observation(1, 35)
        obs['state_continuous'] = torch.randn(1, 35, 18)
        # 只有 3 帧真实前导，其余右侧 padding 应被移动到 TCN 窗口左端。
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
            empty = model(obs, 31, torch.tensor([0]))
            restarted, _ = model.step_logits({key: array[:, 31] for key, array in obs.items()})
            self.assertTrue(torch.allclose(empty[:, 0], restarted, atol=1e-5))

    def test_tcn_never_calls_gru_and_bypassed_weights_do_not_change(self):
        learner = Learner(tcn_config())
        before = {name: value.clone() for name, value in learner.model.gru.state_dict().items()}
        batch = {'observation': tensor_observation(1, 35), 'burn_lengths': torch.tensor([31]),
                 'joint_action_id': torch.full((1, 4), 192, dtype=torch.long), 'mask': torch.ones(1, 4, dtype=torch.bool)}
        with patch.object(learner.model.gru, 'forward', side_effect=AssertionError('TCN 不得调用 GRU')):
            result = learner.train_batch(batch, diagnostics=True)
        self.assertEqual(result['module_changes']['gru'], 0)
        self.assertEqual(result['module_gradients']['gru'], 0)
        self.assertGreater(result['module_changes']['tcn'], 0)
        self.assertTrue(learner.model.module_status()['gru']['bypassed'])
        for name, value in learner.model.gru.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_same_seed_preserves_shared_random_initialization(self):
        config = tcn_config()
        gru = copy.deepcopy(config)
        gru['model']['temporal_mode'] = 'gru'
        left, right = Learner(gru), Learner(config)
        for name, value in left.model.state_dict().items():
            self.assertTrue(torch.equal(value, right.model.state_dict()[name]), name)

    def test_freeze_is_not_bypass_in_gru_mode(self):
        config = tcn_config()
        config['model']['temporal_mode'] = 'gru'
        config['training']['frozen_modules'] = ['gru']
        model = Learner(config).model
        self.assertTrue(model.module_status()['gru']['frozen'])
        self.assertFalse(model.module_status()['gru']['bypassed'])
        with patch.object(model.gru, 'forward', wraps=model.gru.forward) as forward:
            model(tensor_observation(1, 35), 31, torch.tensor([31]))
            self.assertEqual(forward.call_count, 2)

    def test_checkpoint_modes_and_legacy_gru_are_strict(self):
        config = tcn_config()
        split = {'sha256': 'fixture'}
        norm = {**normalization(), 'split_hash': 'fixture'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'last.pt'
            learner = Learner(config)
            checkpoint.save(path, learner, config, split, norm, 0, 0, 0, None, 0)
            package = checkpoint.load(path)
            self.assertEqual(package['network_version'], TCN_NETWORK_VERSION)
            restored = Learner(config)
            checkpoint.restore(package, restored, split)
            for name, value in learner.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[name]))
            gru_config = copy.deepcopy(config)
            gru_config['model']['temporal_mode'] = 'gru'
            with self.assertRaises(ValueError):
                checkpoint.restore(package, Learner(gru_config), split)
            original = Learner(gru_config)
            checkpoint.save(path, original, gru_config, split, norm, 0, 0, 0, None, 0)
            old = torch.load(path, weights_only=True)
            old['spec'].pop('temporal')
            old['spec']['model'].pop('temporal_mode')
            old['config']['model'].pop('temporal_mode')
            torch.save(old, path)
            legacy = checkpoint.load(path)
            self.assertEqual(legacy['network_version'], NETWORK_VERSION)
            checkpoint.restore(legacy, Learner(gru_config), split)

    def test_mode_validation_and_comparison_conditions(self):
        config = tcn_config()
        validate(config)
        config['training']['burn_in'] = 16
        with self.assertRaisesRegex(ValueError, '31'):
            validate(config)
        config['training']['burn_in'] = 31
        gru = copy.deepcopy(config)
        gru['model']['temporal_mode'] = 'gru'
        self.assertEqual(comparison_conditions(config, 'split', normalization())[1],
                         comparison_conditions(gru, 'split', normalization())[1])
        gru['training']['frozen_modules'] = ['gru']
        self.assertNotEqual(comparison_conditions(config, 'split', normalization())[1],
                            comparison_conditions(gru, 'split', normalization())[1])
        with self.assertRaises(ValueError):
            experiment_path('../unsafe')


if __name__ == '__main__':
    unittest.main()
