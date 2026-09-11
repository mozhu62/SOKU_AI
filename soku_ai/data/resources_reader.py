from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .resources_schema import DATASET_SCHEMA, STATE_FIELDS, CATEGORY_FIELDS, TACTICAL_FIELDS, OBJECT_FIELDS, BUTTON_FIELDS


def read_resource_shard(path: Path, config: dict) -> dict:
    """只读必要数组；卡牌、技能、旧奖励等不参与该版 DQfD，不解压无用字段。"""
    with np.load(path, allow_pickle=False) as source:
        meta = json.loads(str(source["metadata_json"].item()))
        expected = {"dataset_schema": DATASET_SCHEMA, "state_continuous": list(STATE_FIELDS),
                    "state_categorical": list(CATEGORY_FIELDS), "tactical_state": list(TACTICAL_FIELDS),
                    "object_numerical": list(OBJECT_FIELDS), "object_categorical": ["action", "action_block_id"],
                    "button_features": list(BUTTON_FIELDS), "continuous_normalized": False}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{path.name} 的 {key} 不兼容；本入口只接受当前原值 resources_v4 NPZ")
        if meta.get("action_shift") not in (0, 1) or meta["action_shift"] != config["data"]["action_shift"]:
            raise ValueError(f"{path.name} 的 action_shift 与配置不一致，不能重复偏移或静默混用")
        keys = ["state_continuous", "state_categorical", "tactical_state", "episode_id", "transition_valid",
                "terminated", "action_horizontal", "action_vertical", "action_buttons", "action_duration"]
        keys += [f"{side}_object_{suffix}" for side in ("self", "opponent")
                 for suffix in ("numerical", "categorical", "offsets")]
        raw = {key: source[key] for key in keys}
    count = len(raw["episode_id"])
    if count < 2:
        raise ValueError(f"{path.name} 少于两帧")
    shapes = {"state_continuous": (count, 18), "state_categorical": (count, 5),
              "tactical_state": (count, 9), "action_buttons": (count, 6)}
    shapes.update({key: (count,) for key in keys[:10] if key not in shapes})
    for key, shape in shapes.items():
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError(f"{path.name}: {key} 形状或有限性错误")
    for key in ("state_categorical", "episode_id", "action_duration"):
        if not np.issubdtype(raw[key].dtype, np.integer):
            raise ValueError(f"{path.name}: {key} 必须为整数")
    for key in ("tactical_state", "action_buttons", "transition_valid", "terminated"):
        if not np.isin(raw[key], [0, 1]).all():
            raise ValueError(f"{path.name}: {key} 必须为 0/1")
    if any(not np.isin(raw[key], [-1, 0, 1]).all() for key in ("action_horizontal", "action_vertical")):
        raise ValueError(f"{path.name}: 方向轴只能为 -1/0/1")
    if (raw["action_duration"] < 1).any():
        raise ValueError(f"{path.name}: action_duration 必须为正数")
    for side in ("self", "opponent"):
        num, cat, offsets = (raw[f"{side}_object_{suffix}"] for suffix in ("numerical", "categorical", "offsets"))
        if (num.ndim != 2 or num.shape[1] != 8 or cat.shape != (len(num), 2)
                or offsets.shape != (count + 1,) or not np.issubdtype(offsets.dtype, np.integer)
                or not np.issubdtype(cat.dtype, np.integer) or offsets[0] != 0 or offsets[-1] != len(num)
                or not ((np.diff(offsets) >= 0) & (np.diff(offsets) <= 3)).all()
                or not np.isfinite(num).all()):
            raise ValueError(f"{path.name}: {side} 对象数组或最近三个对象约束错误")

    state = raw["state_continuous"].astype(np.float32)
    hp, enemy_hp = state[:, 16], state[:, 17]
    valid = raw["transition_valid"].astype(bool).copy()
    valid[-1] = False
    same_episode = raw["episode_id"][:-1] == raw["episode_id"][1:]
    valid[:-1] &= same_episode
    valid &= (hp > 0) & (enemy_hp > 0)
    valid[1:] &= ~(raw["terminated"][:-1].astype(bool) & same_episode)
    done = raw["terminated"].astype(bool).copy()
    done[:-1] |= (hp[1:] <= 0) | (enemy_hp[1:] <= 0)
    done &= valid

    h, v, buttons = raw["action_horizontal"], raw["action_vertical"], raw["action_buttons"]
    if not np.isin(state[valid, 4], [-1, 1]).all():
        raise ValueError(f"{path.name}: 有效帧缺少可靠朝向，无法转换前进/后退动作")
    data = config["data"]
    facing_right = state[:, 4] > 0 if data["direction_positive_faces_right"] else state[:, 4] < 0
    physical_right = h > 0 if data["horizontal_positive_is_right"] else h < 0
    horizontal = np.where(h == 0, 0, np.where(physical_right == facing_right, 1, 2))
    down = v > 0 if data["vertical_positive_is_down"] else v < 0
    vertical = np.where(v == 0, 0, np.where(down, 2, 1))
    # 源顺序是 melee/dash/light/heavy，DQfD 顺序是 A/B/C/D，不能复用 BC Joint ID。
    abcd = buttons[:, [0, 2, 3, 1]].astype(np.int64)
    actions = ((horizontal * 3 + vertical) * 16 + abcd @ np.array([8, 4, 2, 1])).astype(np.int64)

    rewards = np.zeros(count, np.float32)
    outcomes = np.zeros(count, np.int8)
    outcome = (hp[1:] > 0).astype(np.int8) * (enemy_hp[1:] <= 0) - (enemy_hp[1:] > 0).astype(np.int8) * (hp[1:] <= 0)
    outcomes[:-1] = np.where(done[:-1], outcome, 0)
    reward_cfg = config["reward"]
    dealt = np.maximum(0, enemy_hp[:-1] - np.maximum(enemy_hp[1:], 0))
    taken = np.maximum(0, hp[:-1] - np.maximum(hp[1:], 0))
    rewards[:-1] = (dealt - taken) * float(reward_cfg["damage_scale"])
    rewards += (outcomes > 0) * float(reward_cfg["round_win_reward"])
    rewards += (outcomes < 0) * float(reward_cfg["round_loss_reward"])
    rewards[~valid] = 0

    # NPZ 的 action[t] 已是对齐标签。历史行 t 只能放 action[t-1]，不再应用 action_shift。
    history_valid = np.r_[False, valid[:-1] & ~done[:-1]]
    history = np.zeros((count, 11), np.float32)
    positions = np.flatnonzero(history_valid)
    previous = positions - 1
    history[positions, :2] = np.column_stack((h[previous], v[previous]))
    history[positions, 2:6] = abcd[previous]
    history[positions, 6] = raw["action_duration"][previous]
    history[:, 7:] = state[:, [6, 13, 14, 15]]
    # 每个真实断点重新开始窗口；轨迹内任意采样位置仍保留前面的连续历史。
    episode_start = np.maximum.accumulate(np.where(~history_valid, np.arange(count), 0))
    result = {"state_continuous": np.concatenate((state, raw["tactical_state"].astype(np.float32)), axis=1),
              "state_categorical": raw["state_categorical"].astype(np.int64),
              "history_numerical": history, "history_categorical": raw["state_categorical"][:, :4].astype(np.int64),
              "history_valid": history_valid, "episode_start": episode_start,
              "transition_valid": valid, "actions": actions, "rewards": rewards, "dones": done,
              "round_outcomes": outcomes,
              "ignored_card_frames": np.asarray(np.count_nonzero(buttons[:, 4:].any(axis=1)), np.int64)}
    for side in ("self", "opponent"):
        for suffix, dtype in (("numerical", np.float32), ("categorical", np.int64), ("offsets", np.int64)):
            key = f"{side}_object_{suffix}"
            result[key] = raw[key].astype(dtype, copy=False)
    return result
