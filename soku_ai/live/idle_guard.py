from __future__ import annotations

import torch


class IdleGuard:
    """只限制连续发送的全松键动作，不把受击、硬直或蹲防当作无动作。"""

    def __init__(self) -> None:
        self.interventions = 0
        self.reset()

    def reset(self) -> None:
        self.context = None
        self.start_frame = None
        self.last_frame = None

    def elapsed(self, context: tuple, frame: int) -> int:
        if (context != self.context or self.start_frame is None
                or self.last_frame is None or frame < self.last_frame):
            return 0
        return max(0, frame - self.start_frame)

    def select(self, q_values: torch.Tensor, context: tuple, frame: int,
               enabled: bool, max_frames: int) -> tuple[int, bool]:
        raw = int(q_values[0].argmax().item())
        # DQfD 编号 0 是全松键；不修改原 Q 张量，诊断表继续展示模型原始输出。
        if enabled and raw == 0 and self.elapsed(context, frame) >= max_frames:
            return int(q_values[0, 1:].argmax().item()) + 1, True
        return raw, False

    def record(self, action_id: int, context: tuple, frame: int, overridden: bool) -> None:
        # 仅在发键成功后提交；暂停、过期观测和发送失败不能累计为已执行动作。
        if overridden:
            self.interventions += 1
        if action_id != 0:
            self.reset()
            return
        if (context != self.context or self.start_frame is None
                or self.last_frame is None or frame < self.last_frame):
            self.start_frame = frame
        self.context, self.last_frame = context, frame
