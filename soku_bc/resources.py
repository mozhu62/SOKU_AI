from __future__ import annotations

import numpy as np

from .schema import CARD_NUMERICAL_INDICES, MAX_HAND_CARDS, UNKNOWN_CARD_ID


RESOURCE_SUFFIXES = (
    "skill_valid_mask", "skill_variants", "skill_levels", "skill_effective_levels",
    "card_state", "hand_card_ids", "hand_card_costs", "hand_mask",
)


def validate_player_resources(raw: dict, shape: tuple, label: str) -> None:
    """NPZ 和实战共同校验原始资源；未知槽由 mask 表达，不当成默认技能或卡 ID 0。"""
    tails = {"skill_valid_mask": (), "skill_variants": (4,), "skill_levels": (4,),
             "skill_effective_levels": (4,), "card_state": (8,),
             "hand_card_ids": (MAX_HAND_CARDS,), "hand_card_costs": (MAX_HAND_CARDS,),
             "hand_mask": (MAX_HAND_CARDS,)}
    for key, tail in tails.items():
        value = raw.get(key)
        if (value is None or value.shape != (*shape, *tail)
                or not np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"{label}.{key} 缺失、形状错误或不是整数")
    valid = raw["skill_valid_mask"]
    if ((valid < 0) | (valid > 15)).any():
        raise ValueError(f"{label} 技能有效位超出四个指令槽")
    mask = (valid[..., None] & (1 << np.arange(4))) != 0
    for key, high in (("skill_variants", 2), ("skill_levels", 4), ("skill_effective_levels", 4)):
        value = raw[key]
        if ((value < -1) | (value > high)).any() or (value[~mask] != -1).any():
            raise ValueError(f"{label}.{key} 值域或未知槽标记错误")
    if not np.array_equal(raw["skill_variants"] >= 0, mask) or (raw["skill_levels"][mask] < 0).any():
        raise ValueError(f"{label} 技能类型、学习等级与有效位不一致")
    cards = raw["card_state"]
    if ((cards[..., 6] < 0) | (cards[..., 6] > MAX_HAND_CARDS)).any():
        raise ValueError(f"{label} 手牌数量超出采集槽位")
    if (cards[..., [0, 1, 4]] < 0).any():
        raise ValueError(f"{label} 卡牌能量、数量或费用出现负数")
    ids, costs = raw["hand_card_ids"], raw["hand_card_costs"]
    if ((ids < 0) | (ids > UNKNOWN_CARD_ID) | (costs < 0) | (costs > 65535)).any():
        raise ValueError(f"{label} 卡槽 ID 或费用超出存储范围")
    expected = ((np.arange(MAX_HAND_CARDS) < cards[..., 6, None]) & (ids != UNKNOWN_CARD_ID))
    if not np.isin(raw["hand_mask"], [0, 1]).all() or not np.array_equal(raw["hand_mask"].astype(bool), expected):
        raise ValueError(f"{label} 手牌 mask 与卡槽内容不一致")
    if ((cards[..., 3] != ids[..., 0]) & expected[..., 0]).any() or (
            (cards[..., 4] != costs[..., 0]) & expected[..., 0]).any():
        raise ValueError(f"{label} 当前选中卡与第一个手牌槽不一致")


def player_resource_observation(raw: dict) -> dict[str, np.ndarray]:
    """向量化支持单帧及 [batch, time]；不引入将要执行的动作或未来抽卡信息。"""
    skill_mask = (raw["skill_valid_mask"][..., None].astype(np.int64) & (1 << np.arange(4))) != 0
    categories = np.stack([raw[key] for key in (
        "skill_variants", "skill_levels", "skill_effective_levels")], -1).astype(np.int64)
    # 学习等级 0 编为 1；0 token 专供未知值，避免把未识别当作零级技能。
    categories = np.where(skill_mask[..., None], categories + 1, 0)
    hand_mask = raw["hand_mask"].astype(bool)
    return {
        "skill_categorical": categories,
        "skill_mask": skill_mask,
        "card_numerical": raw["card_state"][..., list(CARD_NUMERICAL_INDICES)].astype(np.float32),
        "hand_card_ids": np.where(hand_mask, raw["hand_card_ids"].astype(np.int64), -1),
        "hand_card_costs": np.where(hand_mask, raw["hand_card_costs"], 0).astype(np.float32),
        "hand_mask": hand_mask,
    }


def normalize_resources(result: dict, norm: dict) -> dict:
    for side in ("self", "opponent"):
        mean, std = norm["cards"]
        key = f"{side}_card_numerical"
        result[key] = np.clip((result[key] - mean) / std, -10, 10).astype(np.float32)
        mean, std = norm["card_cost"]
        key = f"{side}_hand_card_costs"
        result[key] = np.where(result[f"{side}_hand_mask"],
                               np.clip((result[key] - mean[0]) / std[0], -10, 10), 0).astype(np.float32)
    return result


def resource_observation(shard: dict, indices, norm: dict) -> dict[str, np.ndarray]:
    result = {}
    for side in ("self", "opponent"):
        raw = {key: shard[f"{side}_{key}"][indices] for key in RESOURCE_SUFFIXES}
        result.update({f"{side}_{key}": value for key, value in player_resource_observation(raw).items()})
    return normalize_resources(result, norm)


def normalization_arrays(normalization):
    result = {}
    for group, size in (("state", 18), ("objects", 8), ("cards", 5), ("card_cost", 1), ("optional_state", 4)):
        raw = normalization.get(group, {})
        mean, std = (np.asarray(raw.get(key), np.float32) for key in ("mean", "std"))
        if (mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError(f"归一化 schema 不兼容：{group} 缺失或无效；旧 checkpoint 不可用于 joint432")
        result[group] = mean, std
    counts = np.asarray(normalization["optional_state"].get("counts"))
    if counts.shape != (4,) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError("optional_state 有效样本数缺失或无效")
    return result
