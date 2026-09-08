import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from soku_cql.models import CQLNetwork, conservative_gap, selected_q
from soku_cql.config import DEFAULTS, MODEL_DEFAULTS
from soku_cql.checkpoint import load
from soku_cql.learner import Learner
from tests.fixtures import tensor_observation


class JointNetworkTests(unittest.TestCase):
    def test_architecture_sequence_burn_in_and_act(self):
        model = CQLNetwork(MODEL_DEFAULTS).eval()
        obs = tensor_observation()
        q = model(obs, burn_in=2, burn_lengths=torch.tensor([0, 2]))
        self.assertEqual(tuple(q.shape), (2, 3, 432))
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(model.current_encoder.network[0].out_features, 256)
        self.assertEqual(model.current_encoder.previous_action.num_embeddings, 433)
        self.assertEqual(model.current_encoder.previous_action.embedding_dim, 32)
        self.assertEqual(model.fusion[0].in_features, 512)
        self.assertEqual((model.gru.input_size, model.gru.hidden_size, model.gru.num_layers), (256, 128, 1))
        self.assertEqual(model.memory_fusion[0].in_features, 384)
        self.assertEqual(model.joint_head[0].out_features, 128)
        self.assertEqual(model.joint_head[-1].out_features, 432)
        self.assertEqual(set(model.module_groups()), {"current_encoder", "object_encoder", "fusion", "gru", "memory_fusion", "joint_head"})
        frame = {key: value[:, 0] for key, value in obs.items()}
        step_q, memory = model.step_q(frame)
        action, act_memory = model.act(frame)
        torch.testing.assert_close(action, step_q.argmax(-1))
        torch.testing.assert_close(memory, act_memory)
        self.assertEqual(tuple(memory.shape), (2, 128))

    def test_conservative_term_over_all_joint_actions(self):
        q = torch.full((2, 3, 432), 4.0, requires_grad=True)
        labels = torch.tensor([[0, 192, 431], [12, 243, 60]])
        data = selected_q(q, labels)
        gap = conservative_gap(q, data, 0.7)
        torch.testing.assert_close(gap, torch.full((2, 3), 0.7 * math.log(432)))
        gap.mean().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertNotEqual(float(q.grad[0, 0, 1]), 0)

    def test_double_dqn_selects_online_action_and_respects_mask(self):
        class FixedNetwork(nn.Module):
            def __init__(self, values, trainable):
                super().__init__()
                self.q = nn.Parameter(values, requires_grad=trainable)

            def forward(self, obs, burn_in, lengths):
                return self.q

        learner = Learner.__new__(Learner)
        learner.config = copy.deepcopy(DEFAULTS)
        learner.config["training"]["n_step"] = 1
        learner.amp, learner.device = False, torch.device("cpu")
        online = torch.zeros(1, 3, 432)
        online[0, 1, 7], online[0, 2, 9] = 10, 10
        target = torch.zeros_like(online)
        target[0, 1, 7], target[0, 1, 8] = 3, 100
        target[0, 2, 9] = 4
        learner.online, learner.target = FixedNetwork(online, True), FixedNetwork(target, False)
        batch = {"observation": {}, "burn_lengths": torch.zeros(1, dtype=torch.long),
                 "joint_action_id": torch.tensor([[0, 1]]), "n_step_returns": torch.tensor([[2.0, 999.0]]),
                 "n_step_steps": torch.tensor([[1, 0]]), "bootstrap_indices": torch.tensor([[1, 2]]),
                 "bootstrap_discounts": torch.tensor([[0.99, 0.0]]), "mask": torch.tensor([[True, False]])}
        loss, parts = learner.losses(batch)
        torch.testing.assert_close(parts["target"], torch.tensor([2 + 0.99 * 3]))
        self.assertEqual(len(parts["q"]), 1)
        loss.backward()
        self.assertIsNone(learner.target.q.grad)
        self.assertTrue(torch.isfinite(learner.online.q.grad).all())
        batch["bootstrap_discounts"][0, 0] = 0
        _, parts = learner.losses(batch)
        torch.testing.assert_close(parts["target"], torch.tensor([2.0]))

    def test_metrics_perfect_prediction_and_constant_target(self):
        labels = torch.tensor([0, 192, 431])
        values = torch.tensor([1.0, 2.0, 3.0])
        joint = torch.zeros(3, 432).scatter(-1, labels[:, None], values[:, None])
        parts = {"q": values, "target": values, "gap": torch.ones(3), "td_loss": torch.tensor(0.0),
                 "joint_q": joint, "joint_labels": labels, "n_step_steps": torch.tensor([5, 3, 1]),
                 "n_step_full": torch.tensor([True, False, False]),
                 "bootstrap_active": torch.tensor([True, True, False]),
                 "expert_imitation_loss": torch.tensor(2.0),
                 "expert_imitation_contribution": torch.tensor(0.02)}
        result = Learner.metrics(parts)
        self.assertEqual(result["td_mse"], 0)
        self.assertEqual(result["td_mae"], 0)
        self.assertEqual(result["ev"], 1)
        self.assertEqual(result["joint_accuracy"], 1)
        self.assertAlmostEqual(result["neutral_data_fraction"], 1/3)
        self.assertEqual(len(result["joint_pred"]), 432)
        self.assertEqual(result["n_step_mean"], 3)
        self.assertAlmostEqual(result["n_step_full_fraction"], 1/3)
        self.assertAlmostEqual(result["bootstrap_fraction"], 2/3)
        self.assertEqual(result["expert_imitation_loss"], 2)
        self.assertAlmostEqual(result["expert_imitation_contribution"], 0.02)
        self.assertIn("action", result["joint_data_top"][0])
        parts["target"] = torch.ones(3)
        self.assertIsNone(Learner.metrics(parts)["ev"])

    def test_legacy_checkpoint_is_rejected_for_every_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"network_version": "soku_cql_recurrent_two_head_resources_v2"}, path)
            with self.assertRaisesRegex(ValueError, "schema 不兼容"):
                load(path)
        with self.assertRaisesRegex(ValueError, "schema 不兼容"):
            CQLNetwork(MODEL_DEFAULTS, "soku_cql_recurrent_two_head_resources_v2")


if __name__ == "__main__":
    unittest.main()
