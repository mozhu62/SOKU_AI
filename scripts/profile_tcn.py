"""隔离 TCN 的正确性、延迟和算子诊断；不启动游戏、不写 checkpoint。"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from soku_bc.checkpoint import load
from soku_bc.models import BCNetwork
from soku_bc.live.streaming_tcn import StreamingTCN
from soku_bc.live.graph_tcn import GraphTCN
from soku_bc.live.batch_graph_candidate import BatchGraphCandidate


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='outputs/tcn_profile')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--batch-graph', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError('iterations 必须为正数')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，无法验证 GPU TCN')
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    package = load(Path(args.checkpoint))
    network = BCNetwork(package['spec']['model'], package['network_version'])
    network.load_state_dict(package['model'], strict=True)
    model = network.tcn.cuda().eval().requires_grad_(False)
    seed = torch.randn(1, model.context_frames, model.input_dim, device='cuda')
    results = []
    with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
        graph = (BatchGraphCandidate if args.batch_graph else GraphTCN)(model, torch.device('cuda:0'), args.amp)
        # 交替长度、未捕获长度回退、跨窗口缓存和重置都要与相同输入的参考路径对照。
        mixed_error = 0.0
        for _ in range(2):
            graph.reset()
            ref = StreamingTCN(model)
            for length in (model.context_frames, 1, 2, 3, 4, 8, 18, 5, 3, 1, 64, 2):
                features = torch.randn(1, length, model.input_dim, device='cuda')
                expected = ref.forward(features)[:, -1:]
                actual = graph.forward(features)
                mixed_error = max(mixed_error, float((expected-actual).abs().max()))
                torch.testing.assert_close(actual, expected, rtol=0.03 if args.amp else 1e-4,
                                           atol=0.03 if args.amp else 1e-4)
        for length in (1, 2, 3, 4, 8, 18):
            chunk = torch.randn(1, length, model.input_dim, device='cuda')
            reference = StreamingTCN(model)
            reference.forward(seed)
            graph.reset()
            graph.forward(seed)
            error = 0.0
            for _ in range(3):
                expected = reference.forward(chunk)[:, -1:]
                actual = graph.forward(chunk)
                error = max(error, float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, rtol=0.03 if args.amp else 1e-4,
                                           atol=0.03 if args.amp else 1e-4)
            for name in ('eager', 'current_graph_router'):
                engine = StreamingTCN(model) if name == 'eager' else graph
                if name != 'eager': engine.reset()
                engine.forward(seed)
                for _ in range(10): engine.forward(chunk)
                torch.cuda.synchronize()
                host, device = [], []
                for _ in range(args.iterations):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before = time.perf_counter()
                    start.record()
                    engine.forward(chunk)
                    end.record()
                    end.synchronize()
                    host.append((time.perf_counter() - before) * 1000)
                    device.append(start.elapsed_time(end))
                row = {'frames': length, 'path': name, 'actual_path': getattr(engine, 'last_path', 'eager'), 'max_abs_error': error,
                       'host_p50_ms': float(np.percentile(host, 50)), 'host_p95_ms': float(np.percentile(host, 95)),
                       'cuda_interval_p50_ms': float(np.percentile(device, 50))}
                results.append(row)
                print(row, flush=True)
                # profiler 与延迟测量分开；trace 可观察转换、复制、kernel 与提交间隙。
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    engine.forward(chunk)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(directory / f'{name}_{length}.json'))
                (directory / f'{name}_{length}.txt').write_text(
                    prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35), encoding='utf-8')
    (directory / 'summary.json').write_text(json.dumps({'checkpoint': args.checkpoint,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'amp': args.amp,
        'batch_graph': args.batch_graph, 'mixed_length_max_abs_error': mixed_error,
        'input': '随机预编码特征，GPU 常驻；不是实战端到端延迟', 'results': results}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
