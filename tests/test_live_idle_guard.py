import torch

from soku_ai.live.idle_guard import IdleGuard


def test_idle_limit_and_reset():
    guard = IdleGuard()
    q = torch.zeros(1, 144)
    q[0, 0], q[0, 48] = 10, 9
    context = (1, 1, "left")
    guard.record(0, context, 100, False)
    assert guard.select(q, context, 159, True, 60) == (0, False)
    assert guard.select(q, context, 160, True, 60) == (48, True)
    assert guard.select(q, context, 160, False, 60) == (0, False)
    assert guard.select(q, (1, 2, "left"), 160, True, 60) == (0, False)
    assert guard.select(q, context, 99, True, 60) == (0, False)
    # 选择本身不代表已发键，只有成功发送后的 record 才提交干预。
    assert guard.interventions == 0
    assert q[0, 0].item() == 10
    guard.record(48, context, 160, True)
    assert guard.interventions == 1
    assert guard.select(q, context, 161, True, 60) == (0, False)
    guard.record(0, context, 162, False)
    guard.reset()
    assert guard.select(q, context, 300, True, 60) == (0, False)


def test_non_neutral_is_not_replaced():
    guard = IdleGuard()
    context = (1, 1, "left")
    guard.record(0, context, 1, False)
    q = torch.zeros(1, 144)
    q[0, 32] = 10
    assert guard.select(q, context, 100, True, 60) == (32, False)
