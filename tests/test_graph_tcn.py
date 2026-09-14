import unittest
import torch
from soku_bc.temporal import TemporalConvEncoder
from soku_bc.live.streaming_tcn import StreamingTCN
from soku_bc.live.graph_tcn import GraphTCN


@unittest.skipUnless(torch.cuda.is_available(), '需要 CUDA 离线验收')
class GraphTCNTests(unittest.TestCase):
    @torch.inference_mode()
    def test_seed_replay_and_reset(self):
        model = TemporalConvEncoder(4, 8, 8, 256).cuda().eval()
        graph = GraphTCN(model, torch.device('cuda:0'), False)
        for _ in range(2):
            graph.reset()
            reference = StreamingTCN(model)
            for length in (256, 1, 3, 2):
                features = torch.randn(1, length, 4, device='cuda')
                expected = reference.forward(features)[:, -1:]
                actual = graph.forward(features)
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
