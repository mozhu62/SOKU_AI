"""符卡模块离线验收源码；不需要启动游戏。"""
import copy
import unittest

import numpy as np
import torch

from soku_bc.config import MODEL_DEFAULTS
from soku_bc.models import BCNetwork
from soku_bc.spells import SpellBranch, spell_loss, SPELL_DATA_VERSION
from soku_bc.spell_data import prepare_spell_data
from soku_bc.live.spell_macro import CardSnapshot, SpellMacro, resolve_action


class SpellTests(unittest.TestCase):
    def test_three_cases_and_weights(self):
        logits = torch.zeros(1, 3, 3, requires_grad=True)
        available = torch.tensor([[[False, False], [True, False], [True, True]]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        loss, parts = spell_loss(logits, torch.tensor([[0, 0, 1]]), available, valid, valid)
        self.assertEqual(parts["mask"].tolist(), [[False, True, True]])
        self.assertEqual(parts["weights"].tolist(), [[0.5, 0.5, 5.0]])
        expected = np.log(3)
        self.assertAlmostEqual(float(loss), expected, places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.0)

    def test_intent_is_not_rejected_by_availability(self):
        logits = torch.tensor([[0., 10., 1.]], requires_grad=True)
        available = torch.zeros(1, 2, dtype=torch.bool)
        valid = torch.ones(1, dtype=torch.bool)
        loss, parts = spell_loss(logits, torch.tensor([1]), available, valid, valid)
        self.assertTrue(parts["mask"].item())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(parts["logits"].argmax(-1)), 1)

    def test_no_available_combat_still_backpropagates(self):
        branch = SpellBranch(8, 2)
        feature = torch.randn(2, 8, requires_grad=True)
        shared, spell = branch(feature, torch.zeros(2, 2, dtype=torch.bool))
        loss, _ = spell_loss(spell, torch.zeros(2, dtype=torch.long), torch.zeros(2, 2, dtype=torch.bool),
                             torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        shared.square().sum().add(loss).backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0)
        self.assertEqual(float(branch.head[0].weight.grad.abs().sum()), 0)

    def test_disabled_checkpoint_keys_unchanged(self):
        torch.manual_seed(5)
        old = BCNetwork(copy.deepcopy(MODEL_DEFAULTS))
        torch.manual_seed(5)
        disabled = BCNetwork(copy.deepcopy(MODEL_DEFAULTS), spell_system={"enabled": False})
        self.assertEqual(old.spec, disabled.spec)
        self.assertEqual(set(old.state_dict()), set(disabled.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, disabled.state_dict()[key]))

    def test_macro_switches_three_times_and_confirms_use(self):
        macro = SpellMacro()
        cards = [200, 201, 202, 203]
        def snapshot(frame, slot, event=0):
            return CardSnapshot((1, 1), frame, tuple(cards), cards[slot], frozenset(cards), True,
                                confirmed_event_serial=event, confirmed_card_id=203 if event else None)
        commands = []
        slot = 0
        for frame in range(11):
            command = resolve_action(64, 203 if frame == 0 else None, snapshot(frame, slot), macro)["kind"]
            commands.append(command)
            if command == "change_card":
                slot += 1
        self.assertEqual(commands.count("change_card"), 3)
        self.assertEqual(commands.count("use_card"), 1)
        macro.tick(snapshot(11, 3, 1))
        self.assertEqual(macro.state, "completed")

    def test_missing_target_and_timeout(self):
        snap = CardSnapshot((1, 1), 0, (200,), 200, frozenset({200}), True)
        macro = SpellMacro(timeout_frames=3)
        self.assertEqual(resolve_action(64, 201, snap, macro)["kind"], "combat")
        resolve_action(64, 200, snap, macro)
        for frame in (1, 2, 3):
            macro.tick(CardSnapshot((1, 1), frame, (200,), 200, frozenset({200}), True))
        self.assertEqual(macro.state, "timeout")

    def test_pre_action_alignment_and_round_boundary(self):
        cfg = {"enabled": True, "character_id": 0, "cards": [{"id": 200, "name": "测试卡"}]}
        metadata = {"spell_capture": {"version": SPELL_DATA_VERSION, "character_id": 0,
                    "card_ids": [200], "alignment": "post_state_t_to_used_cards_event_t_plus_1"}}
        shard = {"spell_available_mask": np.array([[1], [0], [1], [1]], bool),
                 "spell_observation_valid": np.ones(4, bool), "spell_event_valid": np.ones(4, bool),
                 "spell_event_card_id": np.array([-1, 200, 200, -1]), "spell_game_frame": np.array([0, 1, 0, 1]),
                 "spell_use_keyframe": np.array([0, 1, 1, 0], bool),
                 "episode_id": np.array([0, 0, 1, 1]), "terminated": np.zeros(4, bool)}
        prepare_spell_data(shard, metadata, np.array([1, 0, 1, 0], bool), cfg)
        self.assertEqual(shard["spell_target"][0], 1)
        self.assertFalse(shard["spell_supervision_mask"][1])
        self.assertTrue(shard["spell_supervision_mask"][2])


if __name__ == "__main__":
    unittest.main()
