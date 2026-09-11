import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from torch.nn import functional as F

from soku_bc import checkpoint
from soku_bc.action_space import START_ACTION_ID, previous_actions
from soku_bc.config import DEFAULTS, load_config, palr_settings
from soku_bc.dataset import read_shard
from soku_bc.keyframes import build_changepoint_mask
from soku_bc.learner import Learner, classification_parts, aggregate_metrics, supervised_previous_actions
from soku_bc.palr import categorical_kernel, feature_rbf_kernel, compute_hscic, sampled_palr
from soku_bc.runtime import Runtime
from tests.fixtures import tensor_observation, raw_shard, normalization


def sample_batch(batch=2, length=16, burn=31):
    rng = torch.Generator().manual_seed(184)
    obs = tensor_observation(batch, burn + length)
    obs["state_continuous"] = torch.randn(batch, burn + length, 18, generator=rng)
    obs["previous_joint_action_id"] = torch.randint(0, 8, (batch, burn + length), generator=rng)
    return {"observation": obs, "burn_lengths": torch.full((batch,), burn, dtype=torch.long),
            "joint_action_id": torch.randint(0, 4, (batch, length), generator=rng),
            "previous_expert_action_id": torch.randint(0, 4, (batch, length), generator=rng),
            "mask": torch.ones(batch, length, dtype=torch.bool)}


def cpu_config(enabled=True):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["training"].update(device="cpu", amp=False, cpu_threads=2)
    cfg["keyframe_weighting"].update(enabled=True, changepoint_weight=4.)
    cfg["palr"].update(enabled=enabled, sample_size=16)
    return cfg


class PalrKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_categorical_kernel_and_id_permutation(self):
        expected = torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        self.assertTrue(torch.equal(categorical_kernel(torch.tensor([10, 10, 20])), expected))
        self.assertTrue(torch.equal(categorical_kernel(torch.tensor([143, 143, 0])), expected))

    def test_official_three_trace_formula(self):
        rng = torch.Generator().manual_seed(5)
        x, y, z = torch.randn(18, 12, generator=rng), torch.arange(18) % 3, torch.arange(18) % 2
        kx, ky, kz = feature_rbf_kernel(x), categorical_kernel(y), categorical_kernel(z)
        w = torch.linalg.solve(kz + 18 * .001 * torch.eye(18), torch.eye(18))
        term1 = kz.T @ w @ (kx * ky) @ w.T @ kz
        term2 = kz.T @ w @ ((kx @ w.T @ kz) * (ky @ w.T @ kz))
        term3 = (kz.T @ w @ kx @ w.T @ kz) * (kz.T @ w @ ky @ w.T @ kz)
        expected = torch.trace(term1 - 2 * term2 + term3) / 18
        torch.testing.assert_close(compute_hscic(x, y, z), expected, atol=2e-5, rtol=1e-4)

    def test_finite_constant_duplicate_and_extreme_features(self):
        for x in (torch.ones(16, 32), torch.zeros(16, 32), torch.eye(16) * 1e30,
                  torch.randn(16, 32, generator=torch.Generator().manual_seed(3))):
            x.requires_grad_()
            loss = compute_hscic(x, torch.arange(16) % 3, torch.arange(16) % 2)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_artificial_leak_is_larger_than_independent_features(self):
        rng = torch.Generator().manual_seed(12)
        y, z = torch.arange(256) % 4, (torch.arange(256) // 4) % 4
        leak = F.one_hot(y, 4).float() + .01 * torch.randn(256, 4, generator=rng)
        independent = torch.randn(256, 4, generator=rng)
        high, low = compute_hscic(leak, y, z), compute_hscic(independent, y, z)
        self.assertGreater(high.item(), low.item() * 3)

    def test_conditioning_is_not_ordinary_hsic(self):
        y = torch.arange(128) % 4
        x = F.one_hot(y, 4).float()
        centering = torch.eye(128) - torch.ones(128, 128) / 128
        hsic = torch.trace(feature_rbf_kernel(x) @ centering @ categorical_kernel(y) @ centering) / 128 ** 2
        conditioned = compute_hscic(x, y, y)
        self.assertGreater(hsic.item(), .01)
        self.assertLess(conditioned.item(), hsic.item() * .1)

    def test_boundary_padding_and_artificial_slice(self):
        actions = np.array([10, 10, 20, 20, 5, 6, 7, 8, 9])
        episodes = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
        valid = np.array([True, True, True, True, False, True, True, True, False])
        terminal = np.array([False, False, False, False, False, False, True, False, False])
        previous, _ = previous_actions(actions, np.ones(9, dtype=int), episodes, valid, terminal)
        supervision = torch.tensor(valid)[None, :]
        changed, eligible = build_changepoint_mask(torch.tensor(actions)[None], supervision, torch.tensor(previous)[None])
        self.assertEqual(eligible.tolist(), [[False, True, True, False, False, False, True, False, False]])
        # 人工切片第一项 t=2 有真实 t=1；不会因为落在 sequence 首项而失效。
        _, sliced = build_changepoint_mask(torch.tensor(actions[2:3])[None], supervision[:, 2:3], torch.tensor(previous[2:3])[None])
        self.assertTrue(sliced.item())
        features = torch.randn(1, 9, 8, requires_grad=True)
        loss, stats = sampled_palr(features, torch.tensor(actions)[None], torch.tensor(previous)[None], supervision, palr_settings())
        self.assertEqual(stats["palr_eligible_samples"], 3)
        loss.backward()
        self.assertEqual(features.grad[~eligible].abs().sum().item(), 0)

    def test_small_sample_skip_has_zero_gradient(self):
        features = torch.randn(1, 2, 8, requires_grad=True)
        loss, stats = sampled_palr(features, torch.tensor([[1, 2]]), torch.tensor([[144, 1]]),
                                   torch.tensor([[True, False]]), palr_settings())
        self.assertTrue(stats["palr_skipped"])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(features.grad.abs().sum().item(), 0)

    def test_amp_kernel_is_float32(self):
        x = torch.randn(12, 16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = compute_hscic(x, torch.arange(12) % 3, torch.arange(12) % 2)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 来验证真实 fp16 AMP")
    def test_cuda_amp_kernel_gradient(self):
        x = torch.randn(64, 256, device="cuda", dtype=torch.float16, requires_grad=True)
        actions = torch.arange(64, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = compute_hscic(x, actions % 4, actions % 3)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)


class PalrIntegrationTests(unittest.TestCase):
    def test_disabled_loss_gradient_and_rng_are_exactly_original(self):
        learner = Learner(cpu_config(False))
        batch = sample_batch()
        before_rng = torch.get_rng_state().clone()
        total, parts = learner.losses(batch)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertIs(total, parts["loss_keyframe_bc"])
        logits = learner.model(batch["observation"], 31, batch["burn_lengths"])
        old, _ = classification_parts(logits, batch["joint_action_id"], batch["mask"],
                                      previous_joint_action_id=supervised_previous_actions(batch, 31),
                                      keyframe_weighting=learner.config["keyframe_weighting"])
        self.assertTrue(torch.equal(total, old))
        params = tuple(learner.model.parameters())
        actual_grad = torch.autograd.grad(total, params)
        expected_grad = torch.autograd.grad(old, params)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad)))

    def test_auxiliary_output_does_not_change_policy_or_parameters(self):
        learner = Learner(cpu_config())
        batch = sample_batch()
        keys = tuple(learner.model.state_dict())
        normal = learner.model(batch["observation"], 31, batch["burn_lengths"])
        logits, aux = learner.model(batch["observation"], 31, batch["burn_lengths"], return_aux=True)
        self.assertTrue(torch.equal(normal, logits))
        self.assertEqual(aux["temporal_feature"].shape, (2, 16, 256))
        self.assertEqual(keys, tuple(learner.model.state_dict()))

    def test_palr_only_backward_reaches_tcn_not_policy_head(self):
        learner = Learner(cpu_config())
        total, parts = learner.losses(sample_batch())
        torch.testing.assert_close(total, parts["loss_keyframe_bc"] + .1 * parts["loss_palr"], rtol=0, atol=0)
        parts["loss_palr"].backward()
        gradients = [p.grad for p in learner.model.tcn.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in learner.model.policy_head.parameters()))

    def test_palr_requires_separate_ground_truth(self):
        learner = Learner(cpu_config())
        batch = sample_batch()
        # 网络输入里有合法历史也不能替代缺失的专家监督。
        batch.pop("previous_expert_action_id")
        with self.assertRaisesRegex(ValueError, "禁止使用 observation"):
            learner.losses(batch)
        batch["previous_expert_action_id"] = torch.full((2, 16), START_ACTION_ID)
        _, parts = learner.losses(batch)
        self.assertEqual(parts["palr_stats"]["palr_eligible_samples"], 0)

    def test_dataset_metadata_is_independent_from_input_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.npz"
            data = raw_shard(10)
            data["action_horizontal"][:] = np.arange(10) % 3 - 1
            np.savez(path, **data)
            shard = read_shard(path, True)
            previous = shard["previous_expert_action_id"].copy()
            np.testing.assert_array_equal(previous[1:9], shard["joint_action_id"][:8])
            self.assertEqual(previous[0], START_ACTION_ID)
            shard["previous_joint_action_id"][:] = START_ACTION_ID
            np.testing.assert_array_equal(previous, shard["previous_expert_action_id"])

    def test_validation_is_deterministic_and_does_not_consume_training_rng(self):
        learner = Learner(cpu_config(False))
        batch = sample_batch()
        rng = torch.get_rng_state().clone()
        before = {k: v.clone() for k, v in learner.model.state_dict().items()}
        first = learner.validate_batch(batch, palr_seed=302)
        second = learner.validate_batch(batch, palr_seed=302)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNotNone(first["validation_hscic"])
        self.assertTrue(all(torch.equal(v, learner.model.state_dict()[k]) for k, v in before.items()))
        empty = sample_batch()
        empty["previous_expert_action_id"].fill_(START_ACTION_ID)
        skipped = learner.validate_batch(empty)
        merged = aggregate_metrics([first, skipped])
        self.assertEqual(merged["validation_hscic"], first["validation_hscic"])
        self.assertEqual(merged["validation_palr_skipped_batches"], 1)

    def test_checkpoint_resumes_palr_sampling_and_old_config_defaults_off(self):
        config = cpu_config()
        learner = Learner(config)
        batch = sample_batch()
        learner.train_batch(batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            split = {"sha256": "palr-fixture"}
            checkpoint.save(path, learner, config, split, normalization(), 1, 1, 32, None, 0)
            expected, parts = learner.losses(batch)
            package = checkpoint.load(path)
            restored = Learner(package["config"])
            checkpoint.restore(package, restored, split)
            actual, restored_parts = restored.losses(batch)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(parts["loss_palr"], restored_parts["loss_palr"]))
            self.assertEqual(restored.config["palr"], config["palr"])
            package["config"].pop("palr")
            torch.save(package, path)
            legacy = checkpoint.load(path)
            self.assertFalse(legacy["config"]["palr"]["enabled"])
            restored.model.load_state_dict(legacy["model"], strict=True)

    def test_diagnostics_fields_and_small_batch_skip(self):
        learner = Learner(cpu_config())
        result = learner.train_batch(sample_batch(), diagnostics=True)
        for key in ("loss_total", "loss_keyframe_bc", "loss_palr", "palr_weighted_loss", "temporal_feature_norm", "tcn_gradient_norm"):
            self.assertTrue(np.isfinite(result[key]), key)
        self.assertEqual(result["palr_sampled_samples"], 16)
        self.assertEqual(result["palr_eligible_samples"], 32)

    def test_runtime_validation_records_hscic_but_selects_nll(self):
        config = cpu_config(False)
        config["training"]["validation_batches"] = 2
        runtime = Runtime(config)
        runtime.learner = Learner(config)
        runtime.store = Mock()
        runtime.store.sample.return_value = sample_batch()
        runtime._record, runtime._save = Mock(), Mock()
        runtime.majority_action_id = 0
        before_rng = torch.get_rng_state().clone()
        runtime._validate()
        row = runtime.snapshot()["latest_validation"]
        self.assertIsNotNone(row["validation_hscic"])
        self.assertEqual(row["selection_metric"], "validation_nll_v1")
        self.assertEqual(runtime.best["value"], row["nll"])
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        runtime._record.assert_called_once()

    def test_four_configs_only_change_palr_and_output(self):
        configs = [load_config(f"configs/bc_palr_{name}.yaml") for name in "abcd"]
        self.assertEqual([c["palr"]["enabled"] for c in configs], [False, True, True, True])
        self.assertEqual([c["palr"]["alpha"] for c in configs[1:]], [.01, .1, 1])
        self.assertEqual(len({c["output"]["directory"] for c in configs}), 4)
        reference = {k: v for k, v in configs[0].items() if k not in ("palr", "output")}
        current = load_config("configs/bc_suika.yaml")
        self.assertEqual(reference, {k: v for k, v in current.items() if k not in ("palr", "output")})
        self.assertTrue(all({k: v for k, v in c.items() if k not in ("palr", "output")} == reference for c in configs))
        for alpha in (0, .001, .01, .1, 1, 10):
            self.assertEqual(palr_settings({"alpha": alpha})["alpha"], alpha)


if __name__ == "__main__":
    unittest.main()
