from __future__ import annotations

import torch

from .utils import move_to_device


class PackedBatchTransfer:
    """按 dtype 合并小张量传输；固定页缓冲有界复用，不缓存整个数据集。"""

    def __init__(self, device):
        self.device = device
        self.buffers = {}
        self.pending = None

    def prepare(self, batch):
        if self.device.type != "cuda":
            return batch, None
        # 只等待上次 H2D，保证不会覆盖 DMA 仍在读取的主机内存；不等待未发生的下一批训练。
        if self.pending is not None:
            self.pending.synchronize()
        groups = {}
        def visit(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, (*path, key))
            else:
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                    raise TypeError("批量搬运只接受 CPU Tensor 字典")
                groups.setdefault(value.dtype, []).append((path, value))
        visit(batch)
        packed = []
        for dtype, items in groups.items():
            count = sum(tensor.numel() for _, tensor in items)
            buffer = self.buffers.get(dtype)
            if buffer is None or buffer.numel() < count:
                buffer = torch.empty(count, dtype=dtype, pin_memory=True)
                self.buffers[dtype] = buffer
            offset, layout = 0, []
            for path, tensor in items:
                size = tensor.numel()
                buffer[offset:offset + size].copy_(tensor.reshape(-1))
                layout.append((path, tensor.shape, offset, size))
                offset += size
            packed.append((buffer[:count], layout))
        return None, packed

    def transfer(self, prepared):
        batch, packed = prepared
        if packed is None:
            return move_to_device(batch, self.device)
        result = {}
        for buffer, layout in packed:
            device_buffer = buffer.to(self.device, non_blocking=True)
            for path, shape, offset, size in layout:
                parent = result
                for key in path[:-1]:
                    parent = parent.setdefault(key, {})
                parent[path[-1]] = device_buffer[offset:offset + size].view(shape)
        if self.pending is None:
            self.pending = torch.cuda.Event()
        self.pending.record(torch.cuda.current_stream(self.device))
        return result
