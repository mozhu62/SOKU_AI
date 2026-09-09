from __future__ import annotations

import numpy as np

from ..resources import player_resource_observation, validate_player_resources
from ..schema import MAX_HAND_CARDS, SKILL_COMMANDS, UNKNOWN_CARD_ID
from .resource_state import matches


def build_resources(payload, frame, player_side):
    if frame is None or not matches(frame, payload):
        raise ValueError("资源版模型需要同一采集帧的技能数据，不能用缺省技能补齐推理输入")
    sides = ("left", "right") if player_side == "left" else ("right", "left")
    observation, summary = {}, {"sample_serial": int(payload.sampleSerial), "battle_frame": int(payload.battleFrame)}
    for side, source in zip(("self", "opponent"), sides):
        player, skills = getattr(payload, source), getattr(frame.resources, source)
        count = int(player.handCount)
        ids = np.asarray([card.id for card in player.handCards], np.int64)
        costs = np.asarray([card.cost for card in player.handCards], np.int64)
        mask = (np.arange(MAX_HAND_CARDS) < count) & (ids != UNKNOWN_CARD_ID)
        raw = {
            "skill_valid_mask": np.asarray(skills.validMask, np.int64),
            "skill_variants": np.asarray([s.variant for s in skills.skills], np.int64),
            "skill_levels": np.asarray([s.learnedLevel for s in skills.skills], np.int64),
            "skill_effective_levels": np.asarray([s.effectiveLevel for s in skills.skills], np.int64),
            "card_state": np.asarray([player.cardGauge, player.cardCount, player.selectedCard,
                                      player.selectedCardData.id, player.selectedCardData.cost,
                                      player.handCapacity, count, player.handCardsUsed], np.int64),
            "hand_card_ids": ids, "hand_card_costs": costs, "hand_mask": mask.astype(np.uint8),
        }
        validate_player_resources(raw, (), side)
        observation.update({f"{side}_{key}": value for key, value in player_resource_observation(raw).items()})
        # 诊断显示的是本次实际输入的原始值，而不是 GUI 刷新时另取的一帧。
        summary[side] = {
            "skills": [{"command": command, "valid": bool(skills.validMask & (1 << index)),
                        "variant": int(s.variant), "learned_level": int(s.learnedLevel),
                        "effective_level": int(s.effectiveLevel)}
                       for index, (command, s) in enumerate(zip(SKILL_COMMANDS, skills.skills))],
            "card_gauge": int(player.cardGauge), "card_count": int(player.cardCount), "hand_count": count,
            "hand_selected_first": [{"slot": index, "id": int(ids[index]) if mask[index] else None,
                                     "cost": int(costs[index]) if mask[index] else None,
                                     "selected": index == 0} for index in range(count)],
        }
    return observation, summary
