"""严格的符卡采集扩展协议：当前可用性是输入，下一真实帧确认事件是标签。"""
import numpy as np

from .spells import SPELL_DATA_VERSION, system_settings


RAW_FIELDS = ("spell_available_mask", "spell_observation_valid", "spell_event_card_id",
              "spell_event_valid", "spell_game_frame", "spell_use_keyframe")


def prepare_spell_data(shard, metadata, valid, settings):
    cfg = system_settings(settings)
    card_ids = [card["id"] for card in cfg["cards"]]
    expected = {"version": SPELL_DATA_VERSION, "character_id": cfg["character_id"],
                "card_ids": card_ids, "alignment": "post_state_t_to_used_cards_event_t_plus_1"}
    if metadata.get("spell_capture") != expected or any(key not in shard for key in RAW_FIELDS):
        raise ValueError("缺少确认用卡/发动前可用性协议，旧 NPZ 不能启用符卡监督；不能从扣卡或用卡按键猜测标签")
    n = len(valid)
    available = shard["spell_available_mask"]
    obs_valid, event_valid = shard["spell_observation_valid"], shard["spell_event_valid"]
    events, frames = shard["spell_event_card_id"], shard["spell_game_frame"]
    for key in RAW_FIELDS:
        shape = (n, len(card_ids)) if key == "spell_available_mask" else (n,)
        if shard[key].shape != shape:
            raise ValueError(f"符卡字段 {key} 形状错误，期望 {shape}")
    for value in (available, obs_valid, event_valid):
        if value.dtype != np.bool_:
            raise ValueError("符卡可用性/证据有效位必须为 bool")
    if (shard["spell_use_keyframe"].dtype != np.bool_
            or not np.array_equal(shard["spell_use_keyframe"], event_valid & (events >= 0))):
        raise ValueError("用卡 keyframe 必须与有效 usedCards 事件一致")
    if not np.issubdtype(events.dtype, np.integer) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("符卡事件 Card ID 与游戏帧必须为整数")
    if not obs_valid[valid].all():
        # 当前资源同时影响 Combat，未知不能冒充全零可用集合。
        raise ValueError("Combat 有效帧存在未知符卡可用性；请修复采集，不删除整帧或填零继续训练")
    connected = np.zeros(n, bool)
    connected[:-1] = ((shard["episode_id"][:-1] == shard["episode_id"][1:])
                       & (frames[1:] == frames[:-1] + 1))
    # terminated[t] 表示本次转移导致终局；不能屏蔽致胜符卡本身。终局之后由 valid 排除。
    observed = np.zeros(n, bool)
    observed[:-1] = event_valid[1:] & connected[:-1]
    use_ids = np.full(n, -1, np.int64)
    use_ids[:-1] = events[1:]
    if np.any(events[event_valid] < -1):
        raise ValueError("确认事件使用 -1 表示未用卡，其他负数非法")
    target = np.zeros(n, np.int64)
    mapped = use_ids == -1
    for index, card_id in enumerate(card_ids, 1):
        match = use_ids == card_id
        target[match] = index
        mapped |= match
    # 未纳入本次卡表的真实使用不能当成 Active NONE。
    if np.any(valid & observed & ~mapped):
        raise ValueError("usedCards 出现卡表外的真实使用，请补全角色卡表；不能当成 NONE")
    supervision = valid & obs_valid & observed & mapped & (available.any(-1) | (target > 0))
    shard["spell_available_mask"] = available
    shard["spell_target"] = target
    shard["spell_supervision_mask"] = supervision
    for key in RAW_FIELDS[1:]:
        shard.pop(key)
