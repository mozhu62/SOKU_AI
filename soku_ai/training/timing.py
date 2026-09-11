from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class TrainingTimings:
    """主机阶段逐步统计；CUDA Event 仅日志步采样，不在每个阶段强制同步。"""

    def __init__(self, device):
        self.device = device
        self.host = defaultdict(float)
        self.active_seconds = 0.0
        self.steps = 0
        self.events = []
        self.sample_gpu = False

    def begin(self, sample_gpu=False):
        self.sample_gpu = sample_gpu and self.device.type == "cuda"
        self.started = time.perf_counter()
        self.events = []

    @contextmanager
    def phase(self, name, gpu=False):
        before = time.perf_counter()
        pair = None
        if gpu and self.sample_gpu:
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record(torch.cuda.current_stream(self.device))
                self.events.append((name, pair))
            self.host[name] += time.perf_counter() - before

    def end(self):
        self.active_seconds += time.perf_counter() - self.started
        self.steps += 1

    def report(self):
        # 调用方已在日志步同步 CUDA；事件读数不把异步 enqueue 时间冒充 GPU 执行耗时。
        result = {f"host_{name}_ms": value * 1000 / max(self.steps, 1) for name, value in self.host.items()}
        result.update({f"gpu_sample_{name}_ms": start.elapsed_time(end) for name, (start, end) in self.events})
        result.update(active_steps_per_second=self.steps / max(self.active_seconds, 1e-9), window_steps=self.steps)
        self.host.clear()
        self.active_seconds = 0.0
        self.steps = 0
        return result
