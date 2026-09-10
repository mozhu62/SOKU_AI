from __future__ import annotations

import torch
from torch import nn

from .embeddings import SafeEmbedding
from .schema import CARD_NUMERICAL_FEATURES, MAX_HAND_CARDS


def resource_output_dim(cfg):
    per_side = (4 * (cfg["skill_embedding_dim"] + 2 * cfg["skill_level_embedding_dim"] + 1)
                + MAX_HAND_CARDS * (cfg["card_embedding_dim"] + 2) + len(CARD_NUMERICAL_FEATURES))
    return 2 * per_side


class SkillSlotEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.variant = nn.Embedding(4, cfg["skill_embedding_dim"], padding_idx=0)
        self.level = nn.Embedding(6, cfg["skill_level_embedding_dim"], padding_idx=0)
        self.effective_level = nn.Embedding(6, cfg["skill_level_embedding_dim"], padding_idx=0)

    def forward(self, value, mask):
        value = torch.where(mask[..., None], value, 0)
        result = torch.cat((self.variant(value[..., 0]), self.level(value[..., 1]),
                            self.effective_level(value[..., 2])), -1)
        return torch.cat((result * mask[..., None], mask[..., None].to(result.dtype)), -1)


class ResourceEncoder(nn.Module):
    """只组织类别 embedding 和原值特征，不再额外增加资源 MLP 或 Fusion 分支。"""
    def __init__(self, cfg):
        super().__init__()
        self.skill_slots = nn.ModuleDict({f"skill_slot_{i}": SkillSlotEmbedding(cfg) for i in range(1, 5)})
        # 所有卡槽和双方共用一张 Card ID 表；槽位次序通过 concat 完整保留。
        self.card = SafeEmbedding(cfg["card_vocab_size"], cfg["card_embedding_dim"])
        self.output_dim = resource_output_dim(cfg)

    def forward(self, obs):
        features = []
        for side in ("self", "opponent"):
            categories, mask = obs[f"{side}_skill_categorical"], obs[f"{side}_skill_mask"]
            skills = [module(categories[..., i, :], mask[..., i])
                      for i, module in enumerate(self.skill_slots.values())]
            hand_mask = obs[f"{side}_hand_mask"]
            cards = self.card(torch.where(hand_mask, obs[f"{side}_hand_card_ids"], -1)) * hand_mask[..., None]
            costs = torch.where(hand_mask, obs[f"{side}_hand_card_costs"], 0)
            hand = torch.cat((cards, costs[..., None], hand_mask[..., None].to(cards.dtype)), -1).flatten(-2)
            features.extend((*skills, obs[f"{side}_card_numerical"], hand))
        return torch.cat(features, -1)
