"""DLL 直接接口到 BC：只编码集合与游戏 usedCards 事件，不推断动作合法性。"""
import numpy as np
from .spells import SPELL_DATA_VERSION

ARCHIVE_SCHEMA = "bc_card_capture_direct_v1"


def resource_card_mask(snapshot, card_ids):
    if len(set(card_ids)) != len(card_ids) or any(not 0 <= card < 65535 for card in card_ids):
        raise ValueError("Card ID 清单必须唯一且合法")
    present = np.arange(16) < snapshot["count"][..., None]
    known = snapshot["valid"].astype(bool)
    ids = snapshot["card_ids"]
    if np.any(present & known[..., None] & ~np.isin(ids, card_ids)):
        raise ValueError("游戏可用集合包含卡表外ID，请补全角色卡表")
    masks = [np.any(present & (ids == card), axis=-1) for card in card_ids]
    result = np.stack(masks, -1) if masks else np.zeros((*snapshot.shape, 0), bool)
    return result & known[..., None], known


def use_evidence(player_frames):
    count = player_frames["event_count"]
    valid = player_frames["event_valid"].astype(bool)
    # 单分类头无法表示同帧多次使用：保留原始列表，但不把多事件训练成 NONE。
    single_valid = valid & (count <= 1)
    # DLL 的 ID 为 uint16；必须先转有符号类型，避免 -1 在 where 中回绕成 65535。
    raw_ids = player_frames["used_card_ids"][..., 0].astype(np.int32)
    ids = np.where(count == 1, raw_ids, np.int32(-1))
    ids[~single_valid] = -1
    return {"used_card_id": ids, "event_valid": single_valid,
            "keyframe": single_valid & (count == 1), "multiple": valid & (count > 1)}


def training_fields(records, side, card_ids, character_id):
    cards = records["card_capture"][:, side]
    available, known = resource_card_mask(cards["after"], card_ids)
    events = use_evidence(cards)
    fields = {"spell_available_mask": available, "spell_observation_valid": known,
              "spell_event_card_id": events["used_card_id"], "spell_event_valid": events["event_valid"],
              "spell_use_keyframe": events["keyframe"],
              "spell_game_frame": records["battle_frame"].astype(np.int64)}
    meta = {"version": SPELL_DATA_VERSION, "character_id": character_id, "card_ids": card_ids,
            "alignment": "post_state_t_to_used_cards_event_t_plus_1"}
    return fields, meta
