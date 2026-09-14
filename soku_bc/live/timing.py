"""主机分段耗时；CUDA 事件低频采样，不在各阶段强制同步。"""
import time
import torch


class InferenceTiming:
    def __init__(self, device, sample=False):
        self.device = device
        self.last = time.perf_counter()
        self.host = {}
        self.events = []
        self.sample = sample and device.type == 'cuda'
        if self.sample:
            self._event('start')

    def _event(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        self.events.append((name, event))

    def mark(self, name):
        now = time.perf_counter()
        self.host[name] = (now - self.last) * 1000
        self.last = now
        if self.sample:
            self._event(name)

    def finish(self):
        gpu = {}
        if self.events:
            # 仅采样推理收尾等待一次；事件区间也可能包含 CPU 提交间隙，不等于纯 kernel 时间。
            self.events[-1][1].synchronize()
            gpu = {name: before.elapsed_time(after)
                   for (_, before), (name, after) in zip(self.events, self.events[1:])}
        return {'host_ms': self.host, 'cuda_timeline_ms': gpu}
