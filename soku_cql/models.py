from __future__ import annotations

import copy

import torch
from torch import nn

from .config import NETWORK_VERSION, MODEL_DEFAULTS
from .action_space import ACTION_SCHEMA, ACTION_COUNT
from .nn_modules import CurrentStateEncoder, ObjectSetEncoder, FusionEncoder, MemoryFusion, initialize
from .schema import policy_input_manifest


class CQLNetwork(nn.Module):
    """逐帧完整 Controller State 的 432-way Q 网络；所有参数从随机初始化开始。"""

    def __init__(self, cfg: dict, network_version: str = NETWORK_VERSION):
        super().__init__()
        if network_version != NETWORK_VERSION:
            raise ValueError(f"CQL checkpoint schema 不兼容：{network_version}；joint432 必须重新随机初始化")
        self.uses_resources = True
        cfg = {**MODEL_DEFAULTS, **cfg}
        for key in ("current_hidden_dim", "object_hidden_dim", "object_set_dim", "fusion_dim", "gru_hidden_dim", "q_hidden_dim"):
            if cfg[key] != MODEL_DEFAULTS[key]:
                raise ValueError(f"joint432 保持主干宽度不变：{key}={MODEL_DEFAULTS[key]}")
        self.spec = {"network_version": network_version, "model": copy.deepcopy(cfg),
                     "inputs": policy_input_manifest(), "action_schema": ACTION_SCHEMA}
        self.current_encoder = CurrentStateEncoder(cfg)
        self.object_encoder = ObjectSetEncoder(cfg)
        self.fusion = FusionEncoder(cfg)
        self.gru = nn.GRU(cfg["fusion_dim"], cfg["gru_hidden_dim"], batch_first=True)
        self.memory_fusion = MemoryFusion(cfg)
        self.joint_head = self._head(cfg, ACTION_COUNT)
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
        nn.init.orthogonal_(self.joint_head[-1].weight, gain=0.01)

    @staticmethod
    def _head(cfg, count):
        return nn.Sequential(nn.Linear(cfg["fusion_dim"], cfg["q_hidden_dim"]), nn.SiLU(),
                             nn.Linear(cfg["q_hidden_dim"], count))

    def module_groups(self):
        return {key: getattr(self, key) for key in
                ("current_encoder", "object_encoder", "fusion", "gru", "memory_fusion", "joint_head")}

    def encode(self, obs):
        current = self.current_encoder(obs)
        objects = [self.object_encoder(obs[f"{side}_object_numerical"], obs[f"{side}_object_categorical"],
                                       obs[f"{side}_object_mask"], side) for side in ("self", "opponent")]
        return self.fusion(torch.cat((current, *objects), -1))

    def forward(self, obs, burn_in: int = 0, burn_lengths=None):
        # 一段输入含 burn-in、学习片段及最后一个 next_state；片段从不跨越断帧或终局。
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
        return self.joint_head(final)

    @torch.no_grad()
    def act(self, obs, memory=None):
        joint_q, memory = self.step_q(obs, memory)
        return joint_q.argmax(-1), memory

    @torch.no_grad()
    def step_q(self, obs, memory=None):
        """单帧推理共享训练网络参数；返回 Q 明细，不把 Q 值误标成动作概率。"""
        feature = self.encode(obs)
        if memory is None:
            memory = feature.new_zeros(feature.shape[0], self.memory_dim)
        _, hidden = self.gru(feature[:, None], memory[None])
        memory = hidden[0]
        final = self.memory_fusion(torch.cat((feature, memory), -1))
        return self.joint_head(final), memory


def selected_q(joint_q, joint_action_id):
    return joint_q.gather(-1, joint_action_id[..., None]).squeeze(-1)


def conservative_gap(joint_q, data_q, temperature):
    # 对全部 432 个完整状态计算保守项，不再假定方向与按钮的 Q 可相加。
    if temperature <= 0:
        raise ValueError("CQL temperature 必须为正数")
    return torch.logsumexp(joint_q / temperature, -1) * temperature - data_q
