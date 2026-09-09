import unittest

import numpy as np
import torch

from soku_bc.action_diagnostics import dataset_action_counts, merge_dataset_counts, prefixed_history_metrics
from soku_bc.action_space import previous_actions, encode, decode
from soku_bc.learner import classification_parts, classification_metrics, diagnostic_previous_actions, aggregate_metrics


class ActionDiagnosticsTests(unittest.TestCase):
    def test_real_history_boundaries_and_invalid_positions(self):
        actions = np.array([192, 192, 200, 200, 200, 200, 220, 220, 220])
        episodes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        valid = np.array([True, True, False, True, True, True, True, True, False])
        terminated = np.array([False, True, False, False, False, True, False, False, False])
        previous, _ = previous_actions(actions, np.ones(9, dtype=int), episodes, valid, terminated)
        # 起点、终局后、断帧后及跨 episode 均不得猜测历史；三处可比较位置全部保持操作。
        np.testing.assert_array_equal(previous, [432, 192, 432, 432, 432, 200, 432, 220, 220])
        counts = dataset_action_counts(actions, previous, valid)
        self.assertEqual(counts, dict(valid_samples=7, previous_action_samples=3,
                                      previous_action_copy_correct=3, action_change_samples=0))
        summary = merge_dataset_counts([counts])
        self.assertEqual(summary['previous_action_baseline'], 1)
        self.assertEqual(summary['action_change_fraction'], 0)

    def test_burn_in_padding_exclusion_and_loss_unchanged(self):
        logits = torch.zeros(1, 4, 432, requires_grad=True)
        labels = torch.tensor([[192, 195, 195, 431]])
        mask = torch.tensor([[True, True, True, False]])
        batch = {'observation': {'previous_joint_action_id': torch.tensor([[30, 31, 432, 192, 195, 2]])},
                 'mask': mask}
        loss, parts = classification_parts(logits, labels, mask)
        gradient_before = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        actual_previous = diagnostic_previous_actions(batch, 2)
        self.assertEqual(actual_previous.tolist(), [432, 192, 195])
        metrics = classification_metrics(parts, actual_previous)
        gradient_after = torch.autograd.grad(loss, logits)[0]
        self.assertTrue(torch.equal(gradient_before, gradient_after))
        self.assertEqual(metrics['previous_action_samples'], 2)
        self.assertEqual(metrics['action_change_samples'], 1)
        self.assertEqual(metrics['previous_action_baseline'], 0.5)
        self.assertAlmostEqual(metrics['action_change_fraction'], 1/3)

    def test_global_accuracy_uses_change_counts_not_batch_averages(self):
        def row(labels, previous, predictions):
            logits = torch.full((1, len(labels), 432), -5.0)
            logits[0, torch.arange(len(labels)), predictions] = 5.0
            loss, parts = classification_parts(logits, torch.tensor([labels]),
                                               torch.ones(1, len(labels), dtype=torch.bool))
            return {**classification_metrics(parts, torch.tensor(previous)), 'loss': float(loss)}
        first = row([2, 2], [1, 2], [2, 2])  # 一个切换帧，正确。
        second = row([3, 3, 3], [1, 1, 1], [1, 1, 1])  # 三个切换帧，均错误。
        third = row([0], [432], [0])  # 起点总体预测正确，但没有可比较历史。
        result = aggregate_metrics([first, second, third])
        self.assertEqual(result['action_change_samples'], 4)
        self.assertEqual(result['action_change_correct'], 1)
        self.assertEqual(result['action_change_accuracy'], 0.25)
        self.assertEqual(result['previous_action_baseline'], 1/5)
        self.assertAlmostEqual(result['action_change_fraction'], 4/6)
        self.assertEqual(prefixed_history_metrics(result, 'val')['val_action_change_accuracy'], 0.25)
        self.assertIsNone(third['action_change_accuracy'])
        self.assertIsNone(third['previous_action_baseline'])

    def test_no_changes_returns_null_accuracy_and_real_zero_fraction(self):
        logits = torch.zeros(1, 2, 432)
        _, parts = classification_parts(logits, torch.tensor([[192, 192]]), torch.ones(1, 2, dtype=torch.bool))
        result = classification_metrics(parts, torch.tensor([192, 192]))
        self.assertIsNone(result['action_change_accuracy'])
        self.assertIsNone(result['action_change_top5_accuracy'])
        self.assertEqual(result['action_change_fraction'], 0)

    def test_synchronized_horizontal_mirror_preserves_change_relation(self):
        actions = np.array([encode(6, 0, 0), encode(6, 2, 0), encode(6, 2, 0), encode(5, 1, 0)])
        previous = np.array([432, actions[0], actions[1], actions[2]])
        def mirror(ids):
            result = ids.copy()
            known = result < 432
            direction, combat, card = decode(result[known])
            mapping = np.array([0, 3, 2, 1, 6, 5, 4, 9, 8, 7])
            result[known] = encode(mapping[direction], combat, card)
            return result
        valid = np.ones(len(actions), bool)
        self.assertEqual(dataset_action_counts(actions, previous, valid),
                         dataset_action_counts(mirror(actions), mirror(previous), valid))


if __name__ == '__main__':
    unittest.main()
