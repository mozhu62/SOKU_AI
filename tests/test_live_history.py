import unittest

from soku_cql.live.input_history import ControllerHistory
from soku_cql.action_space import encode


class LiveHistoryTests(unittest.TestCase):
    def test_default_shift_uses_consumed_input_not_predicted_action(self):
        history = ControllerHistory(1)
        history.observe((1, 0, 10), 6, [0, 0, 0, 0, 0, 0])
        self.assertEqual(history.previous_id, 432)
        history.observe((1, 0, 11), 6, [1, 0, 0, 0, 0, 0])
        self.assertEqual(history.previous_id, encode(6, 1, 0))
        self.assertEqual(history.direction_duration, 2)
        history.observe((1, 0, 11), 6, [1, 0, 0, 0, 0, 0])
        self.assertEqual(history.direction_duration, 2)
        # 按钮变化不重置方向持续时间；缺帧、切局、用户暂停则重置。
        history.observe((1, 0, 12), 6, [0, 1, 0, 0, 0, 0])
        self.assertEqual(history.direction_duration, 3)
        history.observe((1, 0, 14), 6, [0, 0, 0, 0, 0, 0])
        self.assertEqual(history.previous_id, 432)
        history.observe((1, 1, 15), 6, [0, 0, 0, 0, 0, 0])
        self.assertEqual(history.previous_id, 432)
        history.reset()
        self.assertEqual(history.previous_id, 432)

    def test_shift_zero_reads_prior_frame(self):
        history = ControllerHistory(0)
        history.observe((1, 0, 10), 4, [1, 0, 0, 0, 0, 0])
        history.observe((1, 0, 11), 6, [0, 1, 0, 0, 0, 0])
        self.assertEqual(history.previous_id, encode(4, 1, 0))
        self.assertAlmostEqual(history.previous_duration, 1/60)

    def test_card_overlap_uses_same_cleaning_as_offline_history(self):
        for shift in (0, 1):
            history = ControllerHistory(shift)
            history.observe((1, 0, 10), 6, [0, 0, 0, 0, 0, 0])
            history.observe((1, 0, 11), 6, [1, 1, 0, 0, 1, 1])
            if shift == 0:
                history.observe((1, 0, 12), 6, [0, 0, 0, 0, 0, 0])
            self.assertEqual(history.previous_id, encode(6, 3, 2))
            self.assertAlmostEqual(history.previous_duration, 2/60)


if __name__ == "__main__":
    unittest.main()
