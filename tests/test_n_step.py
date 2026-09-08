import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from soku_cql.checkpoint import load
from soku_cql.config import DEFAULTS, NETWORK_VERSION, validate
from soku_cql.action_space import ACTION_SCHEMA
from soku_cql.schema import policy_input_manifest
from soku_cql.dataset import ReplayStore, read_shard
from soku_cql.learner import Learner
from soku_cql.n_step import build_n_step_targets, target_spec
from tests.fixtures import raw_shard


class NStepReturnTests(unittest.TestCase):
    def build(self, rewards, *, terminal=None, positions=(0,), ends=None, length=2, n=3, gamma=0.5):
        rewards = np.asarray(rewards, np.float32)
        terminal = np.zeros(len(rewards), bool) if terminal is None else np.asarray(terminal, bool)
        ends = [len(rewards) - 1] * len(positions) if ends is None else ends
        return build_n_step_targets(rewards, terminal, np.asarray(positions), np.asarray(ends), length, n, gamma)

    def test_full_discounted_return_and_bootstrap_indices(self):
        row = self.build([1, 2, 3, 4, 5, 6, 0])
        np.testing.assert_allclose(row["n_step_returns"], [[2.75, 4.5]])
        np.testing.assert_array_equal(row["n_step_steps"], [[3, 3]])
        np.testing.assert_array_equal(row["bootstrap_indices"], [[3, 4]])
        np.testing.assert_allclose(row["bootstrap_discounts"], [[0.125, 0.125]])

    def test_terminal_includes_last_reward_but_not_later_rewards_or_bootstrap(self):
        row = self.build([1, 2, 999, 999, 0], terminal=[False, True, False, False, False])
        np.testing.assert_allclose(row["n_step_returns"], [[2, 2]])
        np.testing.assert_array_equal(row["n_step_steps"], [[2, 1]])
        np.testing.assert_array_equal(row["bootstrap_indices"], [[2, 2]])
        np.testing.assert_array_equal(row["bootstrap_discounts"], [[0, 0]])

    def test_short_fragment_bootstraps_from_endpoint_and_padding_is_excluded(self):
        # 末状态所在行的 reward/terminated 不属于片段内转移，不能读取进目标。
        row = self.build([1, 2, 999], terminal=[False, False, True], length=4, n=5)
        np.testing.assert_allclose(row["n_step_returns"], [[2, 2, 0, 0]])
        np.testing.assert_array_equal(row["n_step_steps"], [[2, 1, 0, 0]])
        np.testing.assert_array_equal(row["bootstrap_indices"], [[2, 2, 2, 2]])
        np.testing.assert_allclose(row["bootstrap_discounts"], [[0.25, 0.5, 0, 0]])
        np.testing.assert_array_equal(row["mask"], [[True, True, False, False]])

    def test_each_sequence_has_its_own_segment_boundary(self):
        row = self.build([1, 2, 999, 4, 8, 999], positions=(0, 3), ends=(2, 5))
        np.testing.assert_allclose(row["n_step_returns"], [[2, 2], [8, 8]])
        np.testing.assert_array_equal(row["bootstrap_indices"], [[2, 2], [2, 2]])

    def test_n_one_is_single_step_and_gamma_endpoints_are_supported(self):
        row = self.build([1, 2, 0], terminal=[False, True, False], n=1, gamma=0.99)
        np.testing.assert_allclose(row["n_step_returns"], [[1, 2]])
        np.testing.assert_allclose(row["bootstrap_discounts"], [[0.99, 0]])
        np.testing.assert_array_equal(row["bootstrap_indices"], [[1, 2]])
        for gamma, expected_return, expected_discount in ((0.0, 1, 0), (1.0, 6, 1)):
            with self.subTest(gamma=gamma):
                row = self.build([1, 2, 3, 0], length=1, gamma=gamma)
                np.testing.assert_allclose(row["n_step_returns"], [[expected_return]])
                np.testing.assert_allclose(row["bootstrap_discounts"], [[expected_discount]])


class NStepSamplingTests(unittest.TestCase):
    def sample(self, raw, pick, n=5, length=3, split="train"):
        class FixedRng:
            def choice(self, count, size, p):
                return np.zeros(size, np.int64)

            def integers(self, low, high, size):
                return np.full(size, pick, np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, **raw)
            original = path.read_bytes()
            shard = read_shard(path, True)
            store = ReplayStore.__new__(ReplayStore)
            store.split = {"train": ["one"], "validation": ["one"]}
            store.info = {"one": {"transitions": int(shard["segment_cumulative"][-1])}}
            store.get = lambda name: shard
            store.observation = lambda data, indices: {"row": indices,
                "previous_joint_action_id": data["previous_joint_action_id"][indices]}
            cfg = {"batch_size": 1, "sequence_length": length, "burn_in": 2,
                   "replays_per_batch": 1, "n_step": n, "gamma": 0.5}
            result = store.sample(FixedRng(), cfg, split)
            self.assertEqual(path.read_bytes(), original)
            return result, shard

    def test_lookahead_extends_beyond_loss_window_without_extra_labels(self):
        raw = raw_shard(10)
        raw["rewards"][7] = 8
        raw["action_horizontal"][7] = 1
        batch, shard = self.sample(raw, pick=1)
        self.assertEqual(batch["observation"]["row"].shape, (1, 2 + 3 + 5))
        self.assertEqual(batch["joint_action_id"].shape, (1, 3))
        np.testing.assert_array_equal(batch["burn_lengths"], [1])
        np.testing.assert_array_equal(batch["observation"]["row"][0, 2:], np.arange(1, 9))
        np.testing.assert_array_equal(batch["bootstrap_indices"], [[5, 6, 7]])
        np.testing.assert_allclose(batch["n_step_returns"], [[0, 0, 8 * 0.5**4]])
        self.assertEqual(batch["observation"]["previous_joint_action_id"][0, -1], shard["joint_action_id"][7])
        self.assertNotIn("joint_action_id", batch["observation"])

    def test_episode_gap_and_terminal_boundaries(self):
        for boundary in ("episode", "gap", "terminal"):
            with self.subTest(boundary=boundary):
                raw = raw_shard(8)
                raw["rewards"][1], raw["rewards"][3:] = 2, 999
                if boundary == "episode":
                    raw["episode_id"][3:] = 1
                elif boundary == "gap":
                    raw["transition_valid"][2] = False
                else:
                    raw["terminated"][1] = True
                batch, _ = self.sample(raw, pick=1)
                np.testing.assert_array_equal(batch["mask"], [[True, False, False]])
                np.testing.assert_allclose(batch["n_step_returns"], [[2, 0, 0]])
                np.testing.assert_allclose(batch["bootstrap_discounts"], [[0 if boundary == "terminal" else 0.5, 0, 0]])
                self.assertLessEqual(batch["observation"]["row"].max(), 2)

    def test_validation_uses_identical_n_step_sampling_rules(self):
        raw = raw_shard(10)
        raw["rewards"][:] = np.arange(10)
        train, _ = self.sample(raw, 1)
        validation, _ = self.sample(raw, 1, split="validation")
        for key in train:
            if key != "observation":
                np.testing.assert_array_equal(train[key], validation[key])
        single, _ = self.sample(raw, 1, n=1)
        self.assertEqual(single["observation"]["row"].shape, (1, 2 + 3 + 1))
        np.testing.assert_array_equal(single["n_step_returns"], single["rewards"])


class NStepLearnerTests(unittest.TestCase):
    def test_future_double_dqn_state_and_no_gradient_or_cql_on_lookahead(self):
        class FixedNetwork(nn.Module):
            def __init__(self, q, trainable):
                super().__init__()
                self.q = nn.Parameter(q, requires_grad=trainable)

            def forward(self, obs, burn_in, lengths):
                return self.q

        learner = Learner.__new__(Learner)
        learner.config = copy.deepcopy(DEFAULTS)
        learner.config["training"].update(n_step=3, burn_in=2)
        learner.amp, learner.device = False, torch.device("cpu")
        online = torch.zeros(1, 5, 432)
        online[0, 1, 1], online[0, 3, 7], online[0, 4, 9] = 20, 10, 10
        target = torch.zeros_like(online)
        target[0, 1, 1] = 999
        target[0, 3, 7], target[0, 3, 8] = 4, 100
        target[0, 4, 9] = float("nan")
        learner.online, learner.target = FixedNetwork(online, True), FixedNetwork(target, False)
        batch = {"observation": {}, "burn_lengths": torch.tensor([2]), "joint_action_id": torch.tensor([[0, 1]]),
                 "mask": torch.tensor([[True, True]]), "n_step_returns": torch.tensor([[2.75, 3.0]]),
                 "n_step_steps": torch.tensor([[3, 3]]), "bootstrap_indices": torch.tensor([[3, 4]]),
                 "bootstrap_discounts": torch.tensor([[0.125, 0.0]])}
        loss, parts = learner.losses(batch)
        torch.testing.assert_close(parts["target"], torch.tensor([3.25, 3.0]))
        self.assertEqual(parts["joint_q"].shape, (2, 432))
        self.assertFalse(parts["target"].requires_grad)
        loss.backward()
        self.assertIsNone(learner.target.q.grad)
        self.assertTrue(torch.isfinite(learner.online.q.grad).all())
        torch.testing.assert_close(learner.online.q.grad[:, 2:], torch.zeros_like(online[:, 2:]))

    def test_n_step_config_validation(self):
        for n in (0, -1, 121, 1.5, True, "5"):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["training"]["n_step"] = n
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "n_step"):
                validate(cfg)
        for n in (1, 5, 30, 120):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["training"]["n_step"] = n
            self.assertEqual(validate(cfg)["training"]["n_step"], n)

    def test_checkpoint_preserves_old_single_step_semantics_without_touching_file(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["training"].pop("n_step")
        spec = {"network_version": NETWORK_VERSION, "action_schema": ACTION_SCHEMA,
                "inputs": policy_input_manifest()}
        package = {"network_version": NETWORK_VERSION, "spec": spec, "config": cfg,
                   "online": {"weight": torch.tensor([42.0])}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_joint432.pt"
            torch.save(package, path)
            original = path.read_bytes()
            restored = load(path)
            self.assertEqual(restored["config"]["training"]["n_step"], 1)
            torch.testing.assert_close(restored["online"]["weight"], package["online"]["weight"])
            self.assertEqual(path.read_bytes(), original)
            restored["config"]["training"]["n_step"] = 5
            restored["td_target"] = target_spec(restored["config"]["training"])
            torch.save(restored, path)
            self.assertEqual(load(path)["config"]["training"]["n_step"], 5)
            restored["td_target"]["n_step"] = 3
            torch.save(restored, path)
            with self.assertRaisesRegex(ValueError, "TD 目标"):
                load(path)


if __name__ == "__main__":
    unittest.main()
