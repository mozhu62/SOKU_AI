"""DLL Cards.v1 直接接口；只接受和基础状态完全相同的采集序号。"""
import ctypes as ct
import numpy as np
from .shared_state import SharedMemoryClient


class CardSet(ct.Structure):
    _pack_ = 1
    _fields_ = [("valid", ct.c_uint32), ("count", ct.c_uint32), ("cardIds", ct.c_uint16 * 16)]


class PlayerCards(ct.Structure):
    _pack_ = 1
    _fields_ = [("before", CardSet), ("after", CardSet), ("eventValid", ct.c_uint32),
                ("eventCount", ct.c_uint32), ("usedCardIds", ct.c_uint16 * 16)]


class CardShared(ct.Structure):
    _fields_ = [("magic", ct.c_uint32), ("protocolVersion", ct.c_uint32),
                ("structureSize", ct.c_uint32), ("processId", ct.c_uint32),
                ("writeSequence", ct.c_int32), ("battleFrame", ct.c_uint32),
                ("sampleSerial", ct.c_uint64), ("players", PlayerCards * 2)]


class CardMemoryClient(SharedMemoryClient):
    state_type = CardShared
    protocol_magic = 0x53434431
    protocol_version = 1

    def __init__(self, pid):
        self.mapping_name = f"Local\\SokuDataBridge.Cards.v1.{pid}"
        super().__init__()
        if ct.sizeof(CardShared) != 272:
            raise RuntimeError("卡牌共享内存布局不匹配")

    def read_matching(self, state):
        value = self._read_copy()
        if value is None or (value.processId, value.sampleSerial, value.battleFrame) != (
                state.gameProcessId, state.sampleSerial, state.battleFrame):
            return None
        return value


def observation(frame, side, card_ids):
    if side not in ("left", "right"):
        raise ValueError("卡牌视角必须为 left/right")
    cards = frame.players[0 if side == "left" else 1].after
    if cards.valid != 1 or cards.count > 16:
        raise ValueError("当前帧未读到有效可用卡集合，不能默认填零")
    available = set(cards.cardIds[:cards.count])
    if available - set(card_ids):
        raise ValueError("实时可用卡超出模型卡表，不能静默丢弃")
    return {"spell_available_mask": np.asarray([card in available for card in card_ids], dtype=bool)}
