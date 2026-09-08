from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .learner import prepare_cpu_batch


class BatchPrefetch:
    """单后台线程预取后续 CPU 批次；按更新步派生种子，丢弃预取不会改变续训抽样序列。"""

    def __init__(self, store, config, start_step, pin_memory=False):
        self.store = store
        self.config = copy.deepcopy(config)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cql-data")
        self.pending = deque()
        self.next_step = start_step
        self.pin_memory = pin_memory
        for _ in range(config["training"]["prefetch_batches"]):
            self._submit()

    def _submit(self):
        step = self.next_step
        self.next_step += 1
        self.pending.append(self.executor.submit(self._prepare, step))

    def _prepare(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], step, 0]))
        # 固定页内存也在后台准备，训练线程仅进行 non_blocking 设备搬运。
        return prepare_cpu_batch(self.store.sample(rng, self.config["training"]), self.pin_memory)

    def get(self):
        batch = self.pending.popleft().result()
        self._submit()
        return batch

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
