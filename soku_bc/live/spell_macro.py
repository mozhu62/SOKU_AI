"""不依赖键盘 API 的符卡宏状态机；接线前必须具有可信的同帧可用性与确认事件。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CardSnapshot:
    round_key: tuple
    frame: int
    hand_ids: tuple
    selected_id: int | None
    available_ids: frozenset
    observation_valid: bool
    terminal: bool = False
    confirmed_event_serial: int = 0
    confirmed_card_id: int | None = None


class SpellMacro:
    def __init__(self, timeout_frames=45, pulse_interval_frames=3):
        if timeout_frames < 1 or pulse_interval_frames < 2:
            raise ValueError("宏超时必须为正，按键脉冲至少间隔两帧以释放按键")
        self.timeout = timeout_frames
        self.interval = pulse_interval_frames
        self.reset()

    def reset(self, reason="reset"):
        self.target = None
        self.state = reason
        self.last_frame = self.start_frame = self.last_pulse = None
        self.round_key = None
        self.event_serial = 0
        self.waiting_switch = False
        self.selection_before = None
        self.release_pending = False

    def start(self, target, snapshot):
        if self.target is not None:
            return False
        if snapshot.terminal or not snapshot.observation_valid or target not in snapshot.hand_ids:
            self.state = "unavailable"
            return False
        self.target, self.start_frame = target, snapshot.frame
        self.round_key, self.event_serial = snapshot.round_key, snapshot.confirmed_event_serial
        self.last_frame = None
        self.last_pulse = None
        self.waiting_switch = False
        self.state = "selecting"
        return True

    def tick(self, snapshot):
        if self.target is None:
            return "combat"
        if (snapshot.terminal or not snapshot.observation_valid or snapshot.round_key != self.round_key
                or (self.last_frame is not None and snapshot.frame < self.last_frame)):
            self.reset("interrupted")
            return "release"
        if snapshot.frame == self.last_frame:
            return "release"
        self.last_frame = snapshot.frame
        if (snapshot.confirmed_event_serial > self.event_serial
                and snapshot.confirmed_card_id == self.target):
            self.reset("completed")
            return "release"
        if snapshot.frame - self.start_frame >= self.timeout:
            self.reset("timeout")
            return "release"
        if self.target not in snapshot.hand_ids:
            self.reset("target_invalid")
            return "combat"
        if self.release_pending:
            self.release_pending = False
            return "release"
        if self.last_pulse is not None and snapshot.frame - self.last_pulse < self.interval:
            return "release"
        # 每次脉冲后都已释放并重新读取；同 ID 的重复手牌不会卡住切换。
        self.release_pending = True
        self.last_pulse = snapshot.frame
        if snapshot.selected_id == self.target:
            self.state = "await_use_confirmation"
            return "use_card"
        self.selection_before = snapshot.selected_id
        self.waiting_switch = True
        return "change_card"


def resolve_action(combat_action, spell_card_id, snapshot, macro):
    """符卡请求覆盖 Combat；请求失效才回退，不让两个头同时写按键。"""
    if macro.target is None and spell_card_id is not None:
        macro.start(spell_card_id, snapshot)
    command = macro.tick(snapshot)
    return {"kind": command, "combat_action": combat_action if command == "combat" else None,
            "target_card_id": macro.target, "macro_state": macro.state}
