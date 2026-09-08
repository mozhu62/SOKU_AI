import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from soku_cql.config import DEFAULTS, validate
from soku_cql.learner import Learner


class FixedNetwork(nn.Module):
    def __init__(self, q, trainable):
        super().__init__()
        self.q = nn.Parameter(q, requires_grad=trainable)

    def forward(self, observation, burn_in, lengths):
        return self.q


class ExpertImitationTests(unittest.TestCase):
    def fixture(self, weight=0.01, temperature=1.0):
        learner = Learner.__new__(Learner)
        learner.config = copy.deepcopy(DEFAULTS)
        learner.config["training"].update(n_step=1, burn_in=0, cql_alpha=0.0,
                                          expert_imitation_weight=weight, cql_temperature=temperature)
        learner.device, learner.amp = torch.device("cpu"), False
        learner.online = FixedNetwork(torch.zeros(1, 3, 432), True)
        learner.target = FixedNetwork(torch.zeros(1, 3, 432), False)
        batch = {"observation": {}, "burn_lengths": torch.tensor([0]),
                 "joint_action_id": torch.tensor([[243, 431]]), "mask": torch.tensor([[True, False]]),
                 "n_step_returns": torch.zeros(1, 2), "n_step_steps": torch.tensor([[1, 0]]),
                 "bootstrap_indices": torch.tensor([[1, 2]]), "bootstrap_discounts": torch.zeros(1, 2)}
        return learner, batch

    def test_reused_gap_matches_full_action_cross_entropy(self):
        learner, batch = self.fixture(temperature=0.5)
        with torch.no_grad():
            learner.online.q[0, 0] = torch.linspace(-2, 2, 432)
        loss, parts = learner.losses(batch)
        expected = F.cross_entropy(learner.online.q[0, :1] / 0.5, batch["joint_action_id"][0, :1])
        torch.testing.assert_close(parts["expert_imitation_loss"], expected)
        torch.testing.assert_close(parts["expert_imitation_contribution"], 0.01 * expected)
        torch.testing.assert_close(loss, parts["td_loss"] + 0.01 * expected)

    def test_expert_action_gradient_with_no_padding_or_bootstrap_gradient(self):
        learner, batch = self.fixture()
        loss, parts = learner.losses(batch)
        loss.backward()
        # TD 在该构造下梯度为零、CQL alpha=0，只检查微量模仿项的方向和 mask。
        self.assertEqual(float(parts["td_loss"]), 0)
        self.assertLess(float(learner.online.q.grad[0, 0, 243]), 0)
        self.assertGreater(float(learner.online.q.grad[0, 0, 0]), 0)
        torch.testing.assert_close(learner.online.q.grad[:, 1:], torch.zeros(1, 2, 432))
        self.assertIsNone(learner.target.q.grad)
        self.assertNotIn("joint_action_id", batch["observation"])

    def test_zero_disables_contribution_without_changing_td_target(self):
        learner, batch = self.fixture(weight=0.0)
        learner.config["training"]["cql_alpha"] = 1.0
        original = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
        loss, disabled = learner.losses(batch)
        torch.testing.assert_close(loss, disabled["td_loss"] + disabled["gap"].mean())
        self.assertEqual(float(disabled["expert_imitation_contribution"]), 0)
        learner.config["training"]["expert_imitation_weight"] = 0.01
        enhanced, enabled = learner.losses(batch)
        torch.testing.assert_close(enabled["target"], disabled["target"])
        torch.testing.assert_close(enhanced - loss, enabled["expert_imitation_contribution"])
        for key, value in original.items():
            torch.testing.assert_close(batch[key], value)

    def test_shared_temperature_makes_effective_alpha_explicit(self):
        learner, batch = self.fixture(weight=0.01, temperature=0.5)
        learner.config["training"]["cql_alpha"] = 1.0
        with torch.no_grad():
            learner.online.q[0, 0] = torch.linspace(-1, 1, 432)
        loss, parts = learner.losses(batch)
        torch.testing.assert_close(loss, parts["td_loss"] + (1 + 0.01 / 0.5) * parts["gap"].mean())

    def test_weight_validation(self):
        for weight in (-0.01, 1.01, True, "0.01", float("nan"), float("inf")):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["training"]["expert_imitation_weight"] = weight
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "expert_imitation_weight"):
                validate(cfg)
        for weight in (0, 0.01, 1):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["training"]["expert_imitation_weight"] = weight
            self.assertEqual(validate(cfg)["training"]["expert_imitation_weight"], weight)


if __name__ == "__main__":
    unittest.main()
