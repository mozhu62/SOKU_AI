import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soku_cql.dataset import read_shard, ReplayStore
from soku_cql.resources import normalization_arrays
from soku_cql.action_space import from_raw_axes
from soku_cql.config import DEFAULTS
from soku_cql.schema import OPTIONAL_STATE_FEATURES
from tests.fixtures import raw_shard, normalization


class DatasetJointTests(unittest.TestCase):
    def test_normalization_uses_training_only_including_resources(self):
        train, val = raw_shard(), raw_shard()
        train["state_continuous"][:] = 2
        val["state_continuous"][:] = 10000
        for side in ("self", "opponent"):
            train[f"{side}_card_state"][:, 0] = 10
            val[f"{side}_card_state"][:, 0] = 10000
        val["state_optional_continuous"] = np.full((6, 4), 999, np.float32)
        val["state_optional_mask"] = np.ones((6, 4), bool)
        meta = json.loads(str(val["metadata_json"].item()))
        meta["optional_state_continuous"] = list(OPTIONAL_STATE_FEATURES)
        val["metadata_json"] = np.asarray(json.dumps(meta))
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "train.npz", **train)
            np.savez(Path(directory) / "val.npz", **val)
            cfg = copy.deepcopy(DEFAULTS)
            cfg["data"]["directory"] = directory
            split = {"train": ["train.npz"], "validation": ["val.npz"], "files": ["train.npz", "val.npz"], "sha256": "test"}
            store = ReplayStore(cfg, split, lambda message: None)
            np.testing.assert_array_equal(store.normalization["state"]["mean"], np.full(18, 2))
            self.assertEqual(store.normalization["cards"]["mean"][0], 10)
            self.assertFalse(store.optional_enabled.any())
            observation = store.observation(store.get("val.npz"), np.asarray([0]))
            self.assertFalse(observation["state_optional_mask"].any())

    def read(self, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            np.savez(path, **values)
            return read_shard(path, True)

    def test_existing_v4_raw_fields_convert_in_memory(self):
        raw = raw_shard()
        raw["action_buttons"][:, 0] = np.arange(6) % 2
        shard = self.read(raw)
        expected = from_raw_axes(raw["action_horizontal"], raw["action_vertical"], raw["action_buttons"])
        np.testing.assert_array_equal(shard["joint_action_id"], expected)
        np.testing.assert_array_equal(shard["previous_joint_action_id"], np.r_[432, expected[:-1]])
        self.assertFalse(shard["state_optional_mask"].any())
        store = ReplayStore.__new__(ReplayStore)
        store.norm = normalization_arrays(normalization())
        store.optional_enabled = np.zeros(4, bool)
        obs = store.observation(shard, np.asarray([[0, 1, 2]]))
        self.assertNotIn("joint_action_id", obs)
        self.assertNotIn("action_buttons", obs)
        self.assertEqual(obs["previous_joint_action_id"].shape, (1, 3))
        self.assertEqual(obs["self_object_numerical"].shape, (1, 3, 3, 8))
        self.assertEqual(obs["self_hand_card_ids"].shape, (1, 3, 16))

    def test_conflict_at_unused_last_frame_is_not_hidden(self):
        raw = raw_shard()
        raw["action_buttons"][-1, 4:] = 1
        with self.assertRaisesRegex(ValueError, "action schema 不兼容"):
            self.read(raw)

    def test_segments_reset_and_only_joint_training_labels(self):
        raw = raw_shard()
        raw["episode_id"][3:] = 1
        raw["transition_valid"][2] = False
        shard = self.read(raw)
        self.assertEqual(shard["previous_joint_action_id"][3], 432)
        store = ReplayStore.__new__(ReplayStore)
        store.norm = normalization_arrays(normalization())
        store.optional_enabled = np.zeros(4, bool)
        store.split = {"train": ["one"]}
        store.info = {"one": {"transitions": 4}}
        store.get = lambda name: shard
        batch = store.sample(np.random.default_rng(1), {"batch_size": 3, "sequence_length": 3,
                                                       "burn_in": 2, "replays_per_batch": 1})
        self.assertIn("joint_action_id", batch)
        self.assertNotIn("directions", batch)
        self.assertNotIn("buttons", batch)
        self.assertEqual(batch["joint_action_id"].shape, (3, 3))
        self.assertTrue(batch["mask"].any())


if __name__ == "__main__":
    unittest.main()
