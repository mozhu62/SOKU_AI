"""符卡配置、双头共享融合与独立监督；不改变 Combat 动作空间。"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


SYSTEM_DEFAULTS = {"enabled": False, "character_id": 0, "cards": [], "init_combat_from": None,
                   "catalog_file": None}
TRAINING_DEFAULTS = {"active_none_weight": 0.5, "spell_use_weight": 5.0,
                     "max_weight": 10.0, "loss_weight": 1.0}
SPELL_DATA_VERSION = "bc_spell_used_cards_v1"


def system_settings(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(SYSTEM_DEFAULTS):
        raise ValueError("spell_system 包含未知字段")
    cfg = copy.deepcopy({**SYSTEM_DEFAULTS, **settings})
    if cfg['enabled'] and cfg['catalog_file'] and not cfg['cards']:
        from .card_catalog import load_catalog
        from .config import resolve
        cfg['cards'] = load_catalog(resolve(cfg['catalog_file']), cfg['character_id'])
    elif cfg['enabled'] and not cfg['cards']:
        from .card_catalog import static_cards
        cfg['cards'] = static_cards(cfg['character_id'])
    if type(cfg["enabled"]) is not bool or type(cfg["character_id"]) is not int:
        raise ValueError("符卡 enabled 必须为 bool，character_id 必须为整数")
    if cfg["character_id"] < 0 or not isinstance(cfg["cards"], list):
        raise ValueError("符卡角色与卡牌清单无效")
    ids = []
    for card in cfg["cards"]:
        if (not isinstance(card, dict) or set(card) != {"id", "name"}
                or type(card["id"]) is not int or not 0 <= card["id"] < 65535
                or not isinstance(card["name"], str) or not card["name"].strip()):
            raise ValueError("每种卡必须明确提供 id（0..65534）及非空 name")
        ids.append(card["id"])
    if len(ids) != len(set(ids)) or (cfg["enabled"] and not ids):
        raise ValueError("启用符卡必须提供非空、不重复的有序 cards 清单")
    if cfg["init_combat_from"] is not None and not isinstance(cfg["init_combat_from"], str):
        raise ValueError("init_combat_from 必须是旧 BC checkpoint 路径或 null")
    return cfg


def training_settings(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(TRAINING_DEFAULTS):
        raise ValueError("spell_training 包含未知字段")
    cfg = {**TRAINING_DEFAULTS, **settings}
    for key, value in cfg.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"spell_training.{key} 必须为正有限数值")
    if max(cfg["active_none_weight"], cfg["spell_use_weight"]) > cfg["max_weight"]:
        raise ValueError("符卡权重超过 max_weight，请显式调整，不能暗中截断")
    return cfg


def spell_spec(settings):
    cfg = system_settings(settings)
    return {"data_version": SPELL_DATA_VERSION, "character_id": cfg["character_id"],
            "cards": cfg["cards"], "state_dim": len(cfg["cards"]), "embedding_dim": 32,
            "classes": [{"id": None, "name": "NONE"}, *cfg["cards"]],
            "temporal_input": False, "availability_source": "SokuLib.canActivateCard",
            "event_source": "SokuLib.handInfo.usedCards", "output_semantics": "card_intent"}


class SpellBranch(nn.Module):
    def __init__(self, feature_dim, card_count):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(card_count, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.fusion = nn.Linear(feature_dim + 32, feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.SiLU(), nn.Linear(128, card_count + 1))

    def reset_residual(self):
        # 新建分支先不扰动旧 Combat 特征；训练后两个头都能利用当前可用卡资源。
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, feature, available):
        if available.dtype != torch.bool or available.shape != (*feature.shape[:-1], self.encoder[0].in_features):
            raise ValueError("spell_available_mask 必须是与当前特征对齐的 bool [B,L,N] 或 [B,N]")
        resource = self.encoder(available.to(feature.dtype))
        shared = feature + self.fusion(torch.cat((feature, resource), -1))
        return shared, self.head(shared)




def spell_loss(logits, target, available, supervision, valid, settings=None):
    cfg = training_settings(settings)
    if (target.dtype != torch.long or valid.dtype != torch.bool or supervision.dtype != torch.bool
            or target.shape != valid.shape or supervision.shape != valid.shape
            or available.shape[:-1] != valid.shape):
        raise ValueError("符卡标签与全部监督 mask 必须严格同形状")
    if (available.dtype != torch.bool or logits.shape != (*valid.shape, available.shape[-1] + 1)
            or not torch.isfinite(logits).all()):
        raise ValueError("卡牌 logits/可用集合形状、类型或数值非法")
    # 可用集合是输入特征，不是逐帧发动合法性约束；Use 帧即使集合为空仍保留监督。
    masked = logits.float()
    mask = valid & supervision & (available.any(-1) | (target > 0))
    labels = target[mask]
    if torch.any((labels < 0) | (labels >= logits.shape[-1])):
        raise ValueError("有效符卡标签越界")
    weights = torch.where(target > 0, cfg["spell_use_weight"], cfg["active_none_weight"])
    if len(labels):
        selected = masked[mask]
        if not torch.isfinite(selected).all():
            raise ValueError("符卡 logits 含 NaN/Inf")
        ce = F.cross_entropy(selected, labels, reduction="none")
        loss = (ce * weights[mask]).sum() / weights[mask].sum()
    else:
        # 保持可反传的零，但不对无卡帧计算 CE，避免 0 * inf 或误训练 NONE。
        loss = logits.sum() * 0.0
    return loss, {"logits": masked.detach(), "target": target, "mask": mask,
                  "valid": valid, "available": available, "supervision": supervision,
                  "weights": weights, "loss": loss.detach()}


@torch.no_grad()
def spell_counts(parts):
    mask, labels = parts["mask"], parts["target"][parts["mask"]]
    pred = parts["logits"][mask].argmax(-1)
    classes = parts["logits"].shape[-1]
    matrix = torch.bincount(labels * classes + pred, minlength=classes * classes).reshape(classes, classes)
    weight = parts["weights"][mask].sum()
    return {"matrix": matrix.cpu().tolist(), "masked": int((parts["valid"] & ~mask).sum()),
            "weight_sum": float(weight), "weighted_loss_sum": float(parts["loss"] * weight)}


def spell_metrics(counts, cards):
    matrix = torch.tensor(counts["matrix"], dtype=torch.int64)
    true, pred = matrix.sum(1), matrix.sum(0)
    n, uses, predicted = int(true.sum()), int(true[1:].sum()), int(pred[1:].sum())
    use_hits = int(matrix[1:, 1:].sum())
    ratio = lambda a, b: a / b if b else None
    return {"spell_supervised_count": n, "spell_masked_count": counts["masked"],
            "active_none_count": int(true[0]), "spell_use_count": uses,
            "active_none_ratio": ratio(int(true[0]), n), "spell_use_ratio": ratio(uses, n),
            "spell_loss": ratio(counts["weighted_loss_sum"], counts["weight_sum"]),
            "spell_accuracy_all": ratio(int(matrix.diag().sum()), n),
            "spell_accuracy_non_none": ratio(int(matrix.diag()[1:].sum()), uses),
            "spell_use_recall": ratio(use_hits, uses), "spell_use_precision": ratio(use_hits, predicted),
            "predicted_spell_rate": ratio(predicted, n), "expert_spell_rate": ratio(uses, n),
            "spell_cards": [{**card, "sample_count": int(true[i]),
                             "precision": ratio(int(matrix[i, i]), int(pred[i])),
                             "recall": ratio(int(matrix[i, i]), int(true[i]))}
                            for i, card in enumerate(cards, 1)]}
