import unittest

import numpy as np

from soku_cql.action_space import (
    ACTION_COUNT, START_ACTION_ID, NEUTRAL_ACTION_ID, encode, decode,
    to_controller, from_controller, from_raw_axes, action_name, previous_actions,
    clean_card_overlap, validate_buttons, compact_actions,
)


class ActionSchemaTests(unittest.TestCase):
    def test_all_432_round_trips(self):
        seen = set()
        for direction in range(1, 10):
            for combat in range(16):
                for card in range(3):
                    with self.subTest(direction=direction, combat=combat, card=card):
                        action = encode(direction, combat, card)
                        self.assertEqual(action, (direction - 1) * 48 + combat * 3 + card)
                        self.assertEqual(decode(action), (direction, combat, card))
                        d, buttons = to_controller(action)
                        self.assertEqual(from_controller(d, buttons), action)
                        self.assertFalse(buttons[4] and buttons[5])
                        seen.add(action)
        self.assertEqual(seen, set(range(ACTION_COUNT)))

    def test_vectorized_round_trip_and_axes(self):
        ids = np.arange(ACTION_COUNT).reshape(9, 16, 3)
        np.testing.assert_array_equal(encode(*decode(ids)), ids)
        directions, buttons = to_controller(ids)
        h = (directions - 1) % 3 - 1
        v = 1 - (directions - 1) // 3
        np.testing.assert_array_equal(from_raw_axes(h, v, buttons, True), ids)
        np.testing.assert_array_equal(from_raw_axes(h, -v, buttons, False), ids)

    def test_combat_bit_order_and_labels(self):
        for bit in range(4):
            buttons = np.zeros(6, np.int64)
            buttons[bit] = 1
            self.assertEqual(decode(from_controller(5, buttons)), (5, 1 << bit, 0))
        for parts, name in [((5, 0, 0), "5"), ((6, 1, 0), "6+A"), ((6, 3, 0), "6+D+A"),
                            ((2, 4, 0), "2+B"), ((5, 0, 2), "5+USE_CARD"),
                            ((6, 0, 2), "6+USE_CARD"), ((4, 0, 1), "4+CHANGE_CARD")]:
            self.assertEqual(action_name(encode(*parts)), name)
        self.assertEqual(encode(5, 0, 0), NEUTRAL_ACTION_ID)
        self.assertNotEqual(NEUTRAL_ACTION_ID, START_ACTION_ID)

    def test_invalid_values(self):
        for args in ((0, 0, 0), (10, 0, 0), (5, -1, 0), (5, 16, 0), (5, 0, 3), (5.5, 0, 0)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "action schema 不兼容"):
                encode(*args)
        for value in (-1, 432, 1.5):
            with self.assertRaisesRegex(ValueError, "action schema 不兼容"):
                decode(value)
        for buttons in (np.zeros(5), [0, 0, 0, 0, 0, 2], [0, 0, -1, 0, 0, 0], 0):
            with self.subTest(buttons=buttons), self.assertRaisesRegex(ValueError, "六列 0/1"):
                from_controller(5, buttons)

    def test_card_overlap_keeps_use_and_all_combat_bits(self):
        raw = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(np.uint8)
        original = raw.copy()
        cleaned, overlap = clean_card_overlap(raw)
        self.assertEqual(int(overlap.sum()), 16)
        self.assertEqual(cleaned.dtype, raw.dtype)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(cleaned[:, :4], raw[:, :4])
        np.testing.assert_array_equal(cleaned[:, 5], raw[:, 5])
        np.testing.assert_array_equal(cleaned[:, 4], raw[:, 4] & (1 - raw[:, 5]))
        again, repeated = clean_card_overlap(cleaned)
        self.assertFalse(repeated.any())
        np.testing.assert_array_equal(again, cleaned)
        for direction in range(1, 10):
            encoded = from_controller(np.full(64, direction), raw)
            decoded_direction, decoded_buttons = to_controller(encoded)
            np.testing.assert_array_equal(decoded_direction, np.full(64, direction))
            np.testing.assert_array_equal(decoded_buttons, cleaned)
        for dtype in (np.bool_, np.uint8, np.int64, np.float32):
            buttons = np.asarray([1, 1, 0, 0, 1, 1], dtype=dtype)
            self.assertEqual(from_controller(6, buttons), encode(6, 3, 2))
        # 清洗仅用于导入/回读，实际发键仍禁止输出双卡命令。
        with self.assertRaisesRegex(ValueError, "CHANGE_CARD 与 USE_CARD"):
            validate_buttons(raw)
        validate_buttons(cleaned)

    def test_compact_input_cleaning_keeps_axes_duration_and_alignment(self):
        import pandas as pd

        frame = pd.DataFrame({
            "left_input_horizontal": [1, 2, 3, 4], "left_input_vertical": [0, 0, 0, 0],
            "left_input_a": [0, 1, 0, 0], "left_input_d": [0, 1, 0, 0],
            "left_input_b": [0, 0, 0, 0], "left_input_c": [0, 0, 0, 0],
            "left_input_change_card": [1, 2, 0, 1], "left_input_spell_card": [0, 1, 0, 1],
        })
        original = frame.copy(deep=True)
        source = np.asarray([1, 2, 3, 3])
        h, v, duration, buttons = compact_actions(frame, "left", source, np.asarray([0, 0, 0, 1]))
        np.testing.assert_array_equal(h, [1, 1, 1, 1])
        np.testing.assert_array_equal(v, [0, 0, 0, 0])
        np.testing.assert_array_equal(duration, [2, 3, 1, 1])
        np.testing.assert_array_equal(buttons, [[1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])
        pd.testing.assert_frame_equal(frame, original)

    def test_history_has_no_current_label_and_resets(self):
        actions = np.asarray([10, 20, 30, 40, 50, 60, 70])
        duration = np.asarray([1, 100, 3, 4, 5, 6, 7])
        episode = np.asarray([0, 0, 0, 1, 1, 1, 1])
        valid = np.asarray([1, 1, 0, 1, 0, 1, 0], bool)
        terminal = np.zeros(7, bool)
        previous, age = previous_actions(actions, duration, episode, valid, terminal)
        np.testing.assert_array_equal(previous, [432, 10, 20, 432, 40, 432, 60])
        np.testing.assert_allclose(age[:, 0], [0, 1/60, 1, 0, 4/60, 0, 6/60])
        changed = actions.copy()
        changed[1] = 400
        again, _ = previous_actions(changed, duration, episode, valid, terminal)
        self.assertEqual(again[1], previous[1])
        self.assertEqual(again[2], 400)
        terminal[3] = True
        previous, _ = previous_actions(actions, duration, episode, valid, terminal)
        self.assertEqual(previous[4], START_ACTION_ID)


if __name__ == "__main__":
    unittest.main()
