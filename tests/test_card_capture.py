import unittest
import numpy as np
from soku_bc.card_capture import resource_card_mask, use_evidence

SNAPSHOT = np.dtype([("valid", "u4"), ("count", "u4"), ("card_ids", "u2", (16,))])
PLAYER = np.dtype([("before", SNAPSHOT), ("after", SNAPSHOT), ("event_valid", "u4"),
                   ("event_count", "u4"), ("used_card_ids", "u2", (16,))])

class CardTests(unittest.TestCase):
    def test_set_encoding(self):
        snapshots = np.zeros(2, SNAPSHOT)
        snapshots["valid"][0] = 1
        snapshots["count"][0] = 2
        snapshots["card_ids"][0, :2] = [200, 201]
        mask, known = resource_card_mask(snapshots, [201, 200, 202])
        np.testing.assert_array_equal(mask, [[1, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(known, [1, 0])

    def test_none_use_reset_multiple(self):
        rows = np.zeros(4, PLAYER)
        rows["event_valid"] = [1, 1, 0, 1]
        rows["event_count"] = [0, 1, 0, 2]
        rows["used_card_ids"][1, 0] = 200
        result = use_evidence(rows)
        np.testing.assert_array_equal(result["used_card_id"], [-1, 200, -1, -1])
        np.testing.assert_array_equal(result["event_valid"], [1, 1, 0, 0])
        np.testing.assert_array_equal(result["keyframe"], [0, 1, 0, 0])
        self.assertTrue(result["multiple"][3])
