from __future__ import annotations

import copy

import torch
from torch import nn

from .config import NETWORK_VERSION, MODEL_DEFAULTS
from .action_space import ACTION_SCHEMA, ACTION_COUNT
from .nn_modules import CurrentStateEncoder, ObjectSetEncoder, FusionEncoder, MemoryFusion, initialize
from .schema import policy_input_manifest


class BCNetwork(nn.Module):
    """复用 CQL 主干的 432 类行为克隆网络，输出 logits，不输出 Q 或价值。"""

    def __init__(self, cfg: dict, network_version: str = NETWORK_VERSION):
        super().__init__()
        if network_version != NETWORK_VERSION:
            raise ValueError(f"BC checkpoint schema 不兼容：{network_version}；joint432 必须重新随机初始化")
        self.uses_resources = True
        cfg = {**MODEL_DEFAULTS, **cfg}
        for key in ("current_hidden_dim", "object_hidden_dim", "object_set_dim", "fusion_dim", "gru_hidden_dim", "head_hidden_dim"):
            if cfg[key] != MODEL_DEFAULTS[key]:
                raise ValueError(f"joint432 保持主干宽度不变：{key}={MODEL_DEFAULTS[key]}")
        self.spec = {"network_version": network_version, "model": copy.deepcopy(cfg),
                     "inputs": policy_input_manifest(), "action_schema": ACTION_SCHEMA,
                     "output_semantics": "categorical_logits"}
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

    @staticmethod
    def _head(cfg, count):
        return nn.Sequential(nn.Linear(cfg["fusion_dim"], cfg["head_hidden_dim"]), nn.SiLU(),
                             nn.Linear(cfg["head_hidden_dim"], count))

    def module_groups(self):
        return {key: getattr(self, key) for key in
                ("current_encoder", "object_encoder", "fusion", "gru", "memory_fusion", "policy_head")}

    def encode(self, obs):
        current = self.current_encoder(obs)
        objects = [self.object_encoder(obs[f"{side}_object_numerical"], obs[f"{side}_object_categorical"],
                                       obs[f"{side}_object_mask"], side) for side in ("self", "opponent")]
        return self.fusion(torch.cat((current, *objects), -1))

    def forward(self, obs, burn_in: int = 0, burn_lengths=None):
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

    @torch.no_grad()
    def act(self, obs, memory=None):
        logits, memory = self.step_logits(obs, memory)
        return logits.argmax(-1), memory

    @torch.no_grad()
    def step_logits(self, obs, memory=None):
        """返回单帧 logits 和新记忆；概率由调用方使用 softmax(logits) 获得。"""
        feature = self.encode(obs)
        if memory is None:
            memory = feature.new_zeros(feature.shape[0], self.memory_dim)
        _, hidden = self.gru(feature[:, None], memory[None])
        memory = hidden[0]
        final = self.memory_fusion(torch.cat((feature, memory), -1))
        return self.policy_head(final), memory
