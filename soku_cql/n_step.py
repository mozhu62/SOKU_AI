from __future__ import annotations

import numpy as np


TD_TARGET_VERSION = "segment_n_step_double_dqn_v1"


def target_spec(training):
    # 旧 checkpoint 没有 n_step，含义始终是单步，不能套用新训练的默认值。
    return {"td_target_version": TD_TARGET_VERSION,
            "n_step": training.get("n_step", 1), "gamma": training["gamma"]}


def build_n_step_targets(rewards, terminated, positions, ends, length, n_step, gamma):
    """在各自连续片段内构造回报；ends 是末次有效转移的下一状态下标。"""
    offsets = np.arange(length, dtype=np.int64)[None, :]
    transitions = positions[:, None] + offsets
    limits = ends[:, None]
    mask = transitions < limits
    steps = np.zeros(transitions.shape, dtype=np.int64)
    returns = np.zeros(transitions.shape, dtype=np.float64)
    reached_terminal = np.zeros(transitions.shape, dtype=bool)
    for offset in range(n_step):
        active = mask & ~reached_terminal & (transitions + offset < limits)
        indices = np.minimum(transitions + offset, limits)
        returns += (gamma ** offset) * np.where(active, rewards[indices], 0.0)
        steps += active
        reached_terminal |= active & terminated[indices].astype(bool)

    # 非终局短片段仍从已观测到的末状态 bootstrap；断点后的奖励与状态绝不参与。
    discounts = np.where(mask & ~reached_terminal, np.power(gamma, steps), 0.0)
    # 下标相对 burn-in 之后的主序列，不是相对 NPZ，也不包含 burn-in 长度。
    bootstrap_indices = np.minimum(transitions, limits) - positions[:, None] + steps
    return {"n_step_returns": returns.astype(np.float32), "n_step_steps": steps,
            "bootstrap_discounts": discounts.astype(np.float32),
            "bootstrap_indices": bootstrap_indices, "mask": mask}
