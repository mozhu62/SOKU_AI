from __future__ import annotations

import copy

import torch
from torch import nn

from .config import MODEL_DEFAULTS, active_modules, network_version_for
from .action_space import ACTION_SCHEMA, ACTION_COUNT
from .nn_modules import CurrentStateEncoder, ObjectSetEncoder, FusionEncoder, MemoryFusion, initialize
from .schema import policy_input_manifest
from .temporal import TemporalConvEncoder


def network_spec(cfg):
    cfg = {**MODEL_DEFAULTS, **cfg}
    mode = cfg["temporal_mode"]
    return {"network_version": network_version_for(cfg), "model": copy.deepcopy(cfg),
            "inputs": policy_input_manifest(), "action_schema": ACTION_SCHEMA,
            "output_semantics": "categorical_logits",
            "temporal": {"mode": mode, "output_dim": 128, "gru_bypassed": mode == "tcn",
                         "context_frames": 32 if mode == "tcn" else None,
                         "state_semantics": "last_31_battle_features" if mode == "tcn" else "gru_hidden_128"}}


class BCNetwork(nn.Module):
    """复用 CQL 主干的 432 类行为克隆网络，输出 logits，不输出 Q 或价值。"""

    def __init__(self, cfg: dict, network_version: str | None = None):
        super().__init__()
        expected_version = network_version_for(cfg)
        if network_version is not None and network_version != expected_version:
            raise ValueError(f"BC 时序结构不兼容：{network_version}，当前配置需要 {expected_version}；不能跨 GRU/TCN 续训")
        self.uses_resources = True
        cfg = {**MODEL_DEFAULTS, **cfg}
        for key in ("current_hidden_dim", "object_hidden_dim", "object_set_dim", "fusion_dim", "gru_hidden_dim", "head_hidden_dim"):
            if cfg[key] != MODEL_DEFAULTS[key]:
                raise ValueError(f"joint432 保持主干宽度不变：{key}={MODEL_DEFAULTS[key]}")
        self.spec = network_spec(cfg)
        self.temporal_mode = cfg["temporal_mode"]
        self.current_encoder = CurrentStateEncoder(cfg)
        self.object_encoder = ObjectSetEncoder(cfg)
        self.fusion = FusionEncoder(cfg)
        self.gru = nn.GRU(cfg["fusion_dim"], cfg["gru_hidden_dim"], batch_first=True)
        self.memory_fusion = MemoryFusion(cfg)
        self.policy_head = self._head(cfg, ACTION_COUNT)
        self.memory_dim = cfg["gru_hidden_dim"]
        self.apply(initialize)
        # 与 PPO 单层 GRUCell 相同的循环结构，离线整段训练使用融合 GRU 内核。
        for name, parameter in self.gru.named_parameters():
            if "weight_ih" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.xavier_uniform_(gate)
            elif "weight_hh" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        if self.temporal_mode == "tcn":
            # 保留 GRU 仅用于明确冻结/旁路状态；TCN 在公共模块初始化完后创建，
            # 相同 seed 的两组公共编码器、融合层和分类头因此使用相同初始权重。
            self.gru.requires_grad_(False)
            self.tcn = TemporalConvEncoder(cfg["fusion_dim"])
            self.tcn.apply(initialize)

    @staticmethod
    def _head(cfg, count):
        return nn.Sequential(nn.Linear(cfg["fusion_dim"], cfg["head_hidden_dim"]), nn.SiLU(),
                             nn.Linear(cfg["head_hidden_dim"], count))

    def module_groups(self):
        names = ("current_encoder", "object_encoder", "fusion", "gru", "tcn", "memory_fusion", "policy_head")
        return {key: getattr(self, key) for key in names if hasattr(self, key)}

    def module_status(self):
        active = set(active_modules(self.spec["model"]))
        return {name: {"active": name in active,
                       "frozen": not any(p.requires_grad for p in module.parameters()),
                       "bypassed": name not in active}
                for name, module in self.module_groups().items()}

    def encode(self, obs):
        current = self.current_encoder(obs)
        objects = [self.object_encoder(obs[f"{side}_object_numerical"], obs[f"{side}_object_categorical"],
                                       obs[f"{side}_object_mask"], side) for side in ("self", "opponent")]
        return self.fusion(torch.cat((current, *objects), -1))

    def forward(self, obs, burn_in: int = 0, burn_lengths=None):
        if self.temporal_mode == "tcn":
            return self._tcn_forward(obs, burn_in, burn_lengths)
        # 只读取 burn-in 与学习片段；终局、回合和断帧处切断，当前标签从不进入输入。
        features = self.encode({key: value[:, burn_in:] for key, value in obs.items()})
        memory = features.new_zeros(1, features.shape[0], self.memory_dim)
        if burn_in:
            with torch.no_grad():
                prefix = self.encode({key: value[:, :burn_in] for key, value in obs.items()})
                # 前导片段右侧补齐；长度为零的样本严格从零记忆开始，补齐帧不能污染记忆。
                lengths = burn_lengths.cpu().long()
                packed = nn.utils.rnn.pack_padded_sequence(prefix, lengths.clamp_min(1), batch_first=True,
                                                         enforce_sorted=False)
                _, memory = self.gru(packed, memory)
                memory = memory * (lengths > 0).to(memory.device)[None, :, None]
        recurrent, _ = self.gru(features, memory)
        final = self.memory_fusion(torch.cat((features, recurrent), -1))
        return self.policy_head(final)

    def _tcn_forward(self, obs, burn_in, burn_lengths):
        features = self.encode(obs)
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if burn_in:
            if burn_in != 31 or burn_lengths is None:
                raise ValueError("TCN32 训练需要 31 帧上下文和实际历史长度")
            lengths = burn_lengths.to(features.device).long()
            offsets = torch.arange(burn_in, device=features.device)[None] - (burn_in - lengths[:, None])
            prefix_mask = offsets >= 0
            gather = offsets.clamp_min(0)[..., None].expand(-1, -1, features.shape[-1])
            # Dataset 的 GRU 前导历史是右侧 padding；TCN 对齐成左侧 padding，
            # 让最后一个真实历史帧紧邻当前帧，避免短回合起点出现虚假的时间间隔。
            prefix = features[:, :burn_in].gather(1, gather) * prefix_mask[..., None]
            features = torch.cat((prefix, features[:, burn_in:]), dim=1)
            mask[:, :burn_in] = prefix_mask
        temporal = self.tcn(features, mask)[:, burn_in:]
        # 上下文不产生独立 CE，但允许监督帧梯度通过 TCN 回到它实际使用的历史特征。
        final = self.memory_fusion(torch.cat((features[:, burn_in:], temporal), dim=-1))
        return self.policy_head(final)

    @torch.no_grad()
    def act(self, obs, memory=None):
        logits, memory = self.step_logits(obs, memory)
        return logits.argmax(-1), memory

    @torch.no_grad()
    def step_logits(self, obs, memory=None):
        """返回单帧 logits 和新记忆；概率由调用方使用 softmax(logits) 获得。"""
        feature = self.encode(obs)
        if self.temporal_mode == "tcn":
            if memory is not None and (memory.ndim != 3 or memory.shape[0] != feature.shape[0]
                                       or memory.shape[1] > 31 or memory.shape[2] != feature.shape[1]):
                raise ValueError("TCN 实战记忆必须是 [batch,最多31帧,256]，不能传入 GRU 隐藏状态")
            history = feature[:, None] if memory is None else torch.cat((memory, feature[:, None]), dim=1)
            temporal = self.tcn(history)[:, -1]
            final = self.memory_fusion(torch.cat((feature, temporal), dim=-1))
            return self.policy_head(final), history[:, -31:].detach()
        if memory is None:
            memory = feature.new_zeros(feature.shape[0], self.memory_dim)
        _, hidden = self.gru(feature[:, None], memory[None])
        memory = hidden[0]
        final = self.memory_fusion(torch.cat((feature, memory), -1))
        return self.policy_head(final), memory
