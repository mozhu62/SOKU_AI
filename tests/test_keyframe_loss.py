import copy
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from soku_bc.action_space import previous_actions
from soku_bc.config import DEFAULTS, keyframe_weighting_settings, load_config, validate
from soku_bc.keyframes import build_changepoint_mask
from soku_bc.learner import (
    Learner, aggregate_metrics, classification_metrics, classification_parts,
    supervised_previous_actions,
)
from soku_bc.action_diagnostics import KEYFRAME_RATE_KEYS, prefixed_history_metrics


def source_history(actions, episodes=None, valid=None, terminated=None):
    actions = np.asarray(actions, dtype=np.int64)
    size = len(actions)
    episodes = np.zeros(size, dtype=np.int64) if episodes is None else np.asarray(episodes)
    valid = np.ones(size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    terminated = np.zeros(size, dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    previous, _ = previous_actions(actions, np.ones(size, dtype=np.int64), episodes, valid, terminated)
    return torch.tensor(actions), torch.tensor(valid), torch.tensor(previous)


class ChangepointMaskTests(unittest.TestCase):
    def test_requested_sequences(self):
        for actions, expected in (([10, 10, 10, 10], [False, False, False, False]),
                                  ([10, 10, 20, 20, 5], [False, False, True, False, True])):
            with self.subTest(actions=actions):
                changed, eligible = build_changepoint_mask(*source_history(actions))
                self.assertEqual(changed.tolist(), expected)
                self.assertEqual(eligible.tolist(), [False] + [True] * (len(actions) - 1))

    def test_episode_gap_terminal_and_padding_boundaries(self):
        actions, valid, previous = source_history(
            [10, 20, 20, 30, 40, 50, 60, 70],
            episodes=[0, 1, 1, 1, 1, 1, 1, 1],
            # 第 2 个转移无效，沿用 Dataset 规则断开其后历史；最后一项是 padding。
            valid=[True, True, False, True, True, True, True, False],
            terminated=[False, False, False, False, True, False, False, False],
        )
        changed, eligible = build_changepoint_mask(actions, valid, previous)
        self.assertEqual(eligible.tolist(), [False, False, False, False, True, False, True, False])
        self.assertEqual(changed.tolist(), [False, False, False, False, True, False, True, False])

    def test_slice_first_frame_keeps_real_expert_predecessor(self):
        actions, valid, previous = source_history([10, 10, 20, 20, 5])
        changed, eligible = build_changepoint_mask(actions[2:], valid[2:], previous[2:])
        self.assertEqual(changed.tolist(), [True, False, True])
        self.assertTrue(eligible.all())

    def test_supervised_history_skips_burn_in_without_losing_first_frame(self):
        batch = {"observation": {"previous_joint_action_id": torch.tensor([[432, 10, 10, 20, 20]])},
                 "mask": torch.tensor([[True, True, False]])}
        previous = supervised_previous_actions(batch, 2)
        changed, eligible = build_changepoint_mask(torch.tensor([[20, 20, -999]]), batch["mask"], previous)
        self.assertEqual(previous.shape, batch["mask"].shape)
        self.assertEqual(changed.tolist(), [[True, False, False]])
        self.assertEqual(eligible.tolist(), [[True, True, False]])

    def test_shape_dtype_and_padding_tokens(self):
        actions = torch.tensor([[10, 20, -999]])
        mask = torch.tensor([[True, True, False]])
        changed, eligible = build_changepoint_mask(actions, mask, torch.tensor([[432, -1, 20]]))
        self.assertFalse(changed.any())
        self.assertFalse(eligible.any())
        with self.assertRaisesRegex(ValueError, "同形状"):
            build_changepoint_mask(actions, mask, torch.tensor([10, 20, 30]))
        with self.assertRaisesRegex(ValueError, "int64"):
            build_changepoint_mask(actions, mask, torch.tensor([[10., 20., 30.]]))


class KeyframeLossTests(unittest.TestCase):
    def test_requested_weighted_reduction_is_1_point_8(self):
        labels = torch.tensor([[10, 20]])
        probabilities = torch.empty(1, 2, 432)
        for index, ce in enumerate((1., 2.)):
            probability = math.exp(-ce)
            probabilities[0, index].fill_((1. - probability) / 431)
            probabilities[0, index, labels[0, index]] = probability
        loss, parts = classification_parts(
            probabilities.log(), labels, torch.ones_like(labels, dtype=torch.bool),
            previous_joint_action_id=torch.tensor([[10, 10]]),
            keyframe_weighting={"enabled": True, "changepoint_weight": 4},
        )
        torch.testing.assert_close(parts["per_frame_ce"], torch.tensor([[1., 2.]]))
        self.assertEqual(parts["weights"].tolist(), [[1., 4.]])
        self.assertAlmostEqual(loss.item(), 1.8, places=6)

    def test_weight_one_and_disabled_match_old_loss_and_gradients(self):
        generator = torch.Generator().manual_seed(27)
        original = torch.randn(2, 5, 432, generator=generator)
        labels = torch.tensor([[10, 10, 20, -999, -999], [5, 8, 8, 9, 10]])
        previous = torch.tensor([[432, 10, 10, 20, 432], [4, 5, 8, 8, 9]])
        mask = labels >= 0
        for smoothing in (0., .1, .2):
            for settings in ({"enabled": True, "changepoint_weight": 1},
                             {"enabled": False, "changepoint_weight": 16}):
                with self.subTest(smoothing=smoothing, settings=settings):
                    logits = original.clone().requires_grad_()
                    old_loss = F.cross_entropy(logits[mask], labels[mask], label_smoothing=smoothing)
                    new_loss, parts = classification_parts(
                        logits, labels, mask, smoothing, previous_joint_action_id=previous,
                        keyframe_weighting=settings,
                    )
                    torch.testing.assert_close(new_loss, old_loss, rtol=1e-6, atol=1e-7)
                    old_grad = torch.autograd.grad(old_loss, logits, retain_graph=True)[0]
                    new_grad = torch.autograd.grad(new_loss, logits)[0]
                    torch.testing.assert_close(new_grad, old_grad, rtol=1e-6, atol=1e-7)
                    self.assertTrue(torch.equal(parts["weights"], torch.ones_like(parts["weights"])))

    def test_weighted_gradient_shapes_invalid_nan_and_boundary_frames(self):
        labels = torch.tensor([[10, 20, -999], [30, 30, 40]])
        previous = torch.tensor([[432, 10, 20], [432, 30, 30]])
        mask = labels >= 0
        original = torch.randn(2, 3, 432, generator=torch.Generator().manual_seed(42))
        original[~mask] = float("nan")
        for weight in (1, 2, 4, 8, 16):
            with self.subTest(weight=weight):
                logits = original.clone().requires_grad_()
                loss, parts = classification_parts(
                    logits, labels, mask, .1, previous_joint_action_id=previous,
                    keyframe_weighting={"enabled": True, "changepoint_weight": weight},
                )
                expected_weights = torch.tensor([[1., weight, 1.], [1., 1., weight]])
                per_valid = F.cross_entropy(logits[mask], labels[mask], reduction="none", label_smoothing=.1)
                reference = (per_valid * expected_weights[mask]).sum() / expected_weights[mask].sum()
                torch.testing.assert_close(loss, reference)
                for key in ("per_frame_ce", "weights", "is_changepoint", "changepoint_valid_mask", "valid_mask"):
                    self.assertEqual(parts[key].shape, labels.shape)
                torch.testing.assert_close(parts["weights"], expected_weights)
                expected_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
                loss.backward()
                torch.testing.assert_close(logits.grad, expected_gradient)
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertEqual(logits.grad[~mask].abs().sum().item(), 0.)

    def test_missing_history_and_empty_supervision(self):
        logits = torch.zeros(1, 2, 432)
        labels = torch.tensor([[10, 20]])
        with self.assertRaisesRegex(ValueError, "真实上一帧"):
            classification_parts(logits, labels, torch.ones_like(labels, dtype=torch.bool),
                                 keyframe_weighting={"enabled": True, "changepoint_weight": 4})
        with self.assertRaisesRegex(ValueError, "没有有效"):
            classification_parts(logits, labels, torch.zeros_like(labels, dtype=torch.bool))


class KeyframeMetricTests(unittest.TestCase):
    def test_partition_metrics_and_unweighted_overall_metrics(self):
        logits = torch.full((1, 6, 432), -8.)
        labels = torch.tensor([[10, 10, 20, 20, 5, -999]])
        previous = torch.tensor([[432, 10, 10, 20, 20, 9]])
        mask = labels >= 0
        logits[0, torch.arange(6), torch.tensor([10, 10, 10, 5, 5, 0])] = 5.
        logits[0, 2, 20] = 4.  # 一个切换帧 Top-1 错，但真实动作仍在 Top-5。
        _, original = classification_parts(logits, labels, mask)
        loss, weighted = classification_parts(logits, labels, mask, previous_joint_action_id=previous,
                                              keyframe_weighting={"enabled": True, "changepoint_weight": 16})
        before, after = classification_metrics(original), classification_metrics(weighted)
        for key in ("nll", "joint_accuracy", "joint_top5"):
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["hold_top1"], .5)
        self.assertEqual(after["changepoint_top1"], .5)
        self.assertEqual(after["changepoint_top5"], 1.)
        self.assertEqual(after["expert_changepoint_rate"], .5)
        self.assertEqual(after["previous_action_baseline_accuracy"], .5)
        self.assertEqual(after["model_copy_rate"], .5)
        nll = F.cross_entropy(logits[mask], labels[mask], reduction="none")
        self.assertAlmostEqual(after["hold_nll"], nll[[1, 3]].mean().item(), places=6)
        self.assertAlmostEqual(after["changepoint_nll"], nll[[2, 4]].mean().item(), places=6)
        self.assertNotAlmostEqual(loss.item(), after["nll"], places=3)
        prefixed = prefixed_history_metrics(after, "val")
        for key in KEYFRAME_RATE_KEYS:
            self.assertEqual(prefixed[f"val_{key}"], after[key])

    def test_partition_aggregation_uses_group_denominators(self):
        logits = torch.randn(1, 6, 432, generator=torch.Generator().manual_seed(9))
        labels = torch.tensor([[10, 10, 20, 20, 5, 30]])
        previous = torch.tensor([[432, 10, 10, 20, 20, 432]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        _, parts = classification_parts(logits, labels, mask, previous_joint_action_id=previous)
        expected = classification_metrics(parts)
        rows = []
        # 含只有起点、只有保持、只有切换的空分组批次。
        for start, end in ((0, 1), (1, 2), (2, 5), (5, 6)):
            _, partial = classification_parts(logits[:, start:end], labels[:, start:end], mask[:, start:end],
                                              previous_joint_action_id=previous[:, start:end])
            rows.append(classification_metrics(partial))
        self.assertTrue(all(rows[0][key] is None for key in KEYFRAME_RATE_KEYS))
        self.assertIsNone(rows[1]["changepoint_nll"])
        self.assertEqual(rows[1]["expert_changepoint_rate"], 0.)
        result = aggregate_metrics(rows)
        for key in (*KEYFRAME_RATE_KEYS, "nll", "joint_accuracy", "joint_top5"):
            self.assertAlmostEqual(result[key], expected[key], places=5)

    def test_learner_wiring_and_validation_loss_remain_unweighted(self):
        class FixedNetwork(torch.nn.Module):
            def forward(self, observation, burn_in, burn_lengths):
                return observation["test_logits"][:, burn_in:]

        learner = Learner.__new__(Learner)
        learner.config = copy.deepcopy(DEFAULTS)
        learner.config["training"].update(burn_in=2, label_smoothing=.1)
        learner.config["keyframe_weighting"] = {"enabled": True, "changepoint_weight": 8}
        learner.model, learner.device, learner.amp = FixedNetwork(), torch.device("cpu"), False
        batch = {"observation": {
                    "test_logits": torch.randn(1, 5, 432, generator=torch.Generator().manual_seed(2)),
                    "previous_joint_action_id": torch.tensor([[432, 10, 10, 10, 432]])},
                 "joint_action_id": torch.tensor([[10, 20, 30]]), "mask": torch.ones(1, 3, dtype=torch.bool),
                 "burn_lengths": torch.tensor([2])}
        loss, parts = learner.losses(batch)
        self.assertEqual(parts["weights"].tolist(), [[1., 8., 1.]])
        validation = learner.validate_batch(batch)
        expected = F.cross_entropy(batch["observation"]["test_logits"][:, 2:][batch["mask"]],
                                   batch["joint_action_id"][batch["mask"]], label_smoothing=.1)
        self.assertAlmostEqual(validation["loss"], expected.item(), places=5)
        self.assertEqual(validation["expert_changepoint_rate"], .5)
        self.assertNotAlmostEqual(loss.item(), expected.item(), places=3)


class KeyframeConfigTests(unittest.TestCase):
    def test_legacy_config_and_yaml(self):
        old = copy.deepcopy(DEFAULTS)
        old.pop("keyframe_weighting")
        self.assertFalse(validate(old)["keyframe_weighting"]["enabled"])
        for path in ("configs/bc_suika.yaml", "configs/bc_suika_tcn32.yaml"):
            self.assertEqual(load_config(path)["keyframe_weighting"], {"enabled": True, "changepoint_weight": 4.})

    def test_settings_reject_invalid_weights(self):
        for weight in (True, 0, -1, float("nan"), float("inf"), "4"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                keyframe_weighting_settings({"enabled": True, "changepoint_weight": weight})
        for settings in ({"enabled": "false"}, {"unknown": 4}, []):
            with self.assertRaises(ValueError):
                keyframe_weighting_settings(settings)


if __name__ == "__main__":
    unittest.main()
