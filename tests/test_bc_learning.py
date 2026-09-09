import copy
import math
import unittest

import torch
from torch.nn import functional as F

from soku_bc.config import DEFAULTS, MODULES
from soku_bc.learner import Learner, classification_parts, classification_metrics, aggregate_metrics
from soku_bc.models import BCNetwork
from tests.fixtures import tensor_observation


class BCLearningTests(unittest.TestCase):
    def test_cross_entropy_only_and_masked_gradients(self):
        logits = torch.zeros(1, 3, 432, requires_grad=True)
        labels = torch.tensor([[192, 201, -999]])
        mask = torch.tensor([[True, True, False]])
        loss, parts = classification_parts(logits, labels, mask)
        self.assertAlmostEqual(float(loss), math.log(432), places=5)
        self.assertTrue(torch.allclose(loss, F.cross_entropy(logits[mask], labels[mask])))
        loss.backward()
        self.assertLess(logits.grad[0, 0, 192].item(), 0)
        self.assertGreater(logits.grad[0, 0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 2].abs().sum().item(), 0)
        metrics = classification_metrics(parts)
        self.assertEqual(metrics['samples'], 2)
        self.assertAlmostEqual(metrics['normalized_entropy'], 1.0, places=5)
        with self.assertRaisesRegex(ValueError, '没有有效'):
            classification_parts(logits, labels, torch.zeros_like(mask))

    def test_metrics_aggregate_by_valid_frames(self):
        logits = torch.full((1, 3, 432), -8.0)
        logits[:, :, 192] = 8.0
        labels = torch.tensor([[192, 201, 201]])
        rows = []
        for start, end in ((0, 1), (1, 3)):
            loss, parts = classification_parts(logits[:, start:end], labels[:, start:end],
                                               torch.ones(1, end-start, dtype=torch.bool))
            rows.append({**classification_metrics(parts), 'loss': float(loss)})
        result = aggregate_metrics(rows)
        self.assertAlmostEqual(result['joint_accuracy'], 1/3)
        self.assertEqual(result['represented_actions'], 2)
        self.assertAlmostEqual(result['macro_recall'], 0.5)
        self.assertEqual(sum(result['joint_correct']), 1)
        self.assertAlmostEqual(result['nll'], (rows[0]['nll'] + rows[1]['nll']*2)/3)

    def test_architecture_causal_sequence_and_step_inference(self):
        model = BCNetwork(copy.deepcopy(DEFAULTS['model'])).eval()
        obs = tensor_observation(batch=2, length=5)
        self.assertEqual(tuple(model.module_groups()), MODULES)
        self.assertEqual((model.gru.input_size, model.gru.hidden_size, model.gru.num_layers), (256, 128, 1))
        self.assertEqual(model.current_encoder.previous_action.num_embeddings, 433)
        self.assertEqual(model.current_encoder.network[0].out_features, 256)
        self.assertEqual(model.policy_head[-1].out_features, 432)
        with torch.no_grad():
            full = model(obs)
            self.assertEqual(full.shape, (2, 5, 432))
            memory, frames = None, []
            for index in range(5):
                logits, memory = model.step_logits({key: value[:, index] for key, value in obs.items()}, memory)
                frames.append(logits)
            self.assertTrue(torch.allclose(full, torch.stack(frames, 1), atol=1e-5))
            burned = model(obs, 2, torch.tensor([2, 2]))
            self.assertTrue(torch.allclose(full[:, 2:], burned, atol=1e-5))
            changed = {key: value.clone() for key, value in obs.items()}
            changed['state_continuous'][:, 3:] = 10
            self.assertTrue(torch.allclose(full[:, :3], model(changed)[:, :3], atol=1e-5))
            action, _ = model.act({key: value[:, 0] for key, value in obs.items()})
            self.assertTrue(torch.equal(action, full[:, 0].argmax(-1)))

    def test_frozen_backbone_and_classifier_only_update(self):
        config = copy.deepcopy(DEFAULTS)
        config['training'].update(device='cpu', amp=False, burn_in=0,
                                  frozen_modules=[name for name in MODULES if name != 'policy_head'])
        learner = Learner(config)
        self.assertFalse(hasattr(learner, 'target'))
        before = {key: value.clone() for key, value in learner.model.state_dict().items()}
        batch = {'observation': tensor_observation(1, 2), 'burn_lengths': torch.zeros(1, dtype=torch.long),
                 'joint_action_id': torch.full((1, 2), 192, dtype=torch.long),
                 'mask': torch.ones(1, 2, dtype=torch.bool)}
        result = learner.train_batch(batch, diagnostics=True)
        self.assertFalse(result['optimizer_skipped'])
        for key, value in learner.model.state_dict().items():
            if not key.startswith('policy_head.'):
                self.assertTrue(torch.equal(before[key], value), key)
        self.assertGreater(result['module_changes']['policy_head'], 0)
        self.assertTrue(all(result['module_changes'][name] == 0 for name in config['training']['frozen_modules']))


if __name__ == '__main__':
    unittest.main()
