from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    """只在左侧补零；LayerNorm 仅作用于同一帧的通道，不跨时间统计。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.mix = nn.Linear(channels, channels)

    def forward(self, values, mask):
        temporal = self.conv(F.pad(values.transpose(1, 2), (self.dilation, 0))).transpose(1, 2)
        result = F.silu(values + self.mix(F.silu(self.norm(temporal))))
        return result * mask[..., None].to(result.dtype)


class TemporalConvEncoder(nn.Module):
    """五层 kernel=2、dilation=1/2/4/8/16，逐帧感受野恰为 32，不做全段池化。"""

    context_frames = 32
    output_dim = 128

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, self.output_dim),
                                        nn.LayerNorm(self.output_dim), nn.SiLU())
        self.blocks = nn.ModuleList(CausalResidualBlock(self.output_dim, dilation)
                                    for dilation in (1, 2, 4, 8, 16))

    def forward(self, features, mask=None):
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("TCN 有效帧 mask 必须与 [batch,time] 一致")
        # 每层都抹去左侧 padding，防止偏置在假历史上生成可传播的伪特征。
        values = self.projection(features) * mask[..., None].to(features.dtype)
        for block in self.blocks:
            values = block(values, mask)
        return values
