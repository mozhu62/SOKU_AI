from __future__ import annotations

import ctypes as ct

from .shared_state import SharedMemoryClient, StatePayload, FILE_MAP_READ, SharedMemoryProtocolError


CAPACITY = 128


class FrameHeader(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "magic", "protocolVersion", "structureSize", "payloadSize", "resourceSize",
        "capacity", "slotSize", "gameProcessId", "writeSequence", "reserved")]
    _fields_ += [("latestSerial", ct.c_uint64), ("generation", ct.c_uint64)]


class FrameSlot(ct.Structure):
    # DLL LiveFramePublisher.hpp：资源占 104 字节，仅保留传输布局，不作为 DQfD 输入。
    _fields_ = [("writeSequence", ct.c_uint32), ("reserved", ct.c_uint32), ("serial", ct.c_uint64),
                ("payload", StatePayload), ("unusedResources", ct.c_uint32 * 26)]


MAPPING_BYTES = ct.sizeof(FrameHeader) + CAPACITY * ct.sizeof(FrameSlot)


class LiveFrameClient:
    """读取现有 LiveFrames.v1 队列；推理较慢时仍追读中间真实帧。"""

    def __init__(self):
        self.state = SharedMemoryClient()
        self.kernel = self.state._kernel32
        self.kernel.GetTickCount64.argtypes = []
        self.kernel.GetTickCount64.restype = ct.c_uint64
        self.address = self.mapping = self.pid = None
        self.cursor = 0
        self.generation = None
        self.available = False
        self.wait_reason = "等待 LiveFrames.v1 真实帧队列"
        self.reset_reason = None
        self.dropped = 0
        self.latest_tick = 0

    def age_ms(self, payload=None):
        tick = int(payload.publisherTickMs) if payload is not None else self.latest_tick
        now = int(self.kernel.GetTickCount64())
        return now - tick if 0 < tick <= now else float("inf")

    def read_state(self):
        return self.state.read()

    def _header(self):
        pointer = self.address + FrameHeader.writeSequence.offset
        for _ in range(4):
            before = ct.c_uint32.from_address(pointer).value
            if before & 1:
                continue
            header = FrameHeader.from_buffer_copy(ct.string_at(self.address, ct.sizeof(FrameHeader)))
            after = ct.c_uint32.from_address(pointer).value
            if before != after or after & 1:
                continue
            if header.structureSize == 0:
                return None
            actual = tuple(int(getattr(header, key)) for key in (
                "magic", "protocolVersion", "structureSize", "payloadSize", "resourceSize", "capacity", "slotSize", "gameProcessId"))
            expected = (0x534C4631, 1, MAPPING_BYTES, ct.sizeof(StatePayload), 104, CAPACITY, ct.sizeof(FrameSlot), self.pid)
            if (actual != expected or ct.sizeof(FrameHeader) != 56 or FrameSlot.payload.offset != 16
                    or FrameSlot.unusedResources.offset != 16 + ct.sizeof(StatePayload)):
                raise SharedMemoryProtocolError("LiveFrames.v1 布局不匹配，请更新游戏实际加载的 SokuDataBridge.dll")
            return header
        return None

    def _connect(self, pid):
        if self.address is not None and pid == self.pid:
            return True
        self._close_ring()
        mapping = self.kernel.OpenFileMappingW(FILE_MAP_READ, False, f"Local\\SokuDataBridge.LiveFrames.v1.{pid}")
        if not mapping:
            self.wait_reason = "缺少 LiveFrames.v1：请使用已有 BC TCN32 逐帧队列版 SokuDataBridge.dll 并重启游戏"
            return False
        address = self.kernel.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, MAPPING_BYTES)
        if not address:
            self.kernel.CloseHandle(mapping)
            raise SharedMemoryProtocolError("LiveFrames 队列映射失败，请核对 DLL 版本")
        self.mapping, self.address, self.pid = mapping, int(address), pid
        return True

    def _payload(self, serial):
        pointer = self.address + ct.sizeof(FrameHeader) + ((serial - 1) % CAPACITY) * ct.sizeof(FrameSlot)
        for _ in range(4):
            before = ct.c_uint32.from_address(pointer).value
            if before & 1:
                continue
            slot_serial = ct.c_uint64.from_address(pointer + FrameSlot.serial.offset).value
            # 只复制模型需要的状态，跳过不参与本版网络的技能资源尾部。
            raw = ct.string_at(pointer + FrameSlot.payload.offset, ct.sizeof(StatePayload))
            after = ct.c_uint32.from_address(pointer).value
            if before == after and not after & 1 and slot_serial == serial:
                payload = StatePayload.from_buffer_copy(raw)
                if int(payload.gameProcessId) != self.pid:
                    raise SharedMemoryProtocolError("帧队列进程标识不匹配")
                return payload
        return None

    def discard_pending(self):
        if self.address is not None:
            header = self._header()
            if header is not None:
                self.cursor, self.generation = int(header.latestSerial), int(header.generation)

    def read(self):
        self.reset_reason = None
        discovery = self.read_state()
        # seqlock 短暂忙碌不是断帧；保留连接状态，由观测年龄上限负责松键。
        if discovery is None:
            self.wait_reason = "等待 DLL 状态提交"
            return []
        if not discovery.payload.initialized:
            self.available = False
            self.wait_reason = "等待游戏与 SokuDataBridge.dll"
            return []
        if not self._connect(int(discovery.payload.gameProcessId)):
            self.available = False
            return []
        header = self._header()
        if header is None:
            self.wait_reason = "等待 DLL 队列提交"
            return []
        latest = int(header.latestSerial)
        if self.generation != int(header.generation) or latest < self.cursor:
            self.reset_reason = "DLL 队列初始化或游戏进程变化"
            self.generation = int(header.generation)
            self.cursor = max(0, latest - CAPACITY)
        first = max(self.cursor + 1, latest - CAPACITY + 1)
        missed = max(0, first - self.cursor - 1)
        frames = []
        for serial in range(first, latest + 1):
            payload = self._payload(serial)
            if payload is None:
                missed += 1
            else:
                frames.append(payload)
        after = self._header()
        if after is None:
            self.wait_reason = "等待 DLL 队列提交"
            return []
        if int(after.generation) != int(header.generation):
            self.available = False
            self.wait_reason = "DLL 队列重新初始化，等待完整帧"
            return []
        self.available = True
        self.cursor = latest
        self.dropped += missed
        if missed:
            self.reset_reason = f"队列缺失 {missed} 帧，重新累计连续历史"
        if frames:
            self.latest_tick = int(frames[-1].publisherTickMs)
        self.wait_reason = "LiveFrames.v1 已连接"
        return frames

    def _close_ring(self):
        if self.address is not None:
            self.kernel.UnmapViewOfFile(ct.c_void_p(self.address))
        if self.mapping is not None:
            self.kernel.CloseHandle(self.mapping)
        self.mapping = self.address = self.pid = None
        self.available = False
        self.cursor, self.generation, self.latest_tick = 0, None, 0

    def close(self):
        self._close_ring()
        self.state.close()
