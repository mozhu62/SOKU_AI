import unittest

import torch

from soku_bc.temporal import TemporalConvEncoder
from soku_bc.live.streaming_tcn import StreamingTCN


class StreamingTCNTests(unittest.TestCase):
    @torch.inference_mode()
    def test_chunks_and_sliding_window(self):
        for context in (32, 256):
            torch.manual_seed(7)
            model = TemporalConvEncoder(4, 8, 8, context).eval()
            features = torch.randn(1, context + 13, 4)
            expected = model(features)
            stream = StreamingTCN(model)
            start = 0
            for length in (context, 1, 3, 9):
                actual = stream.forward(features[:, start:start + length])
                torch.testing.assert_close(actual, expected[:, start:start + length], atol=2e-5, rtol=2e-5)
                start += length
                # 正常积累满窗口之后，最后输出应与原滑动窗口前向一致。
                torch.testing.assert_close(actual[:, -1], model(features[:, start-context:start])[:, -1],
                                           atol=2e-5, rtol=2e-5)
            self.assertEqual(sum(value.shape[1] for value in stream.cache.values()), context - 1)

    def test_reject_training_mode(self):
        model = TemporalConvEncoder(4, 8, 8)
        with self.assertRaises(ValueError):
            StreamingTCN(model).forward(torch.randn(1, 1, 4))
