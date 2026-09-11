from __future__ import annotations

import torch
from torch import nn

from .embeddings import GameEmbeddings


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = super().forward(values)
        return output if self.causal_padding == 0 else output[:, :, : -self.causal_padding]


class ResidualCausalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.residual(values)
        output = self.dropout(self.activation(self.norm1(self.conv1(values))))
        output = self.dropout(self.activation(self.norm2(self.conv2(output))))
        return self.activation(output + residual)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        numerical_dim: int,
        embeddings: GameEmbeddings,
        model_config: dict,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        input_dim = (
            numerical_dim
            + model_config["action_embedding_dim"] * 2
            + model_config["block_embedding_dim"] * 2
        )
        channels = [int(value) for value in model_config["temporal_channels"]]
        if len(channels) != 4:
            raise ValueError("第一版 TCN 需要四个通道配置，对应 dilation 1/2/4/8")
        self.input_projection = nn.Linear(input_dim, channels[0])
        blocks: list[nn.Module] = []
        in_channels = channels[0]
        for out_channels, dilation in zip(channels, (1, 2, 4, 8), strict=True):
            blocks.append(
                ResidualCausalBlock(
                    in_channels,
                    out_channels,
                    int(model_config["temporal_kernel_size"]),
                    dilation,
                    float(model_config["temporal_dropout"]),
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        output_dim = int(model_config["temporal_output_dim"])
        self.pool_projection = nn.Sequential(
            nn.Linear(channels[-1] * 2, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = torch.cat(
            [
                self.embeddings.action(categorical[:, :, 0]),
                self.embeddings.block(categorical[:, :, 1]),
                self.embeddings.action(categorical[:, :, 2]),
                self.embeddings.block(categorical[:, :, 3]),
            ],
            dim=-1,
        )
        values = self.input_projection(torch.cat([numerical, embedded], dim=-1))
        values = values.transpose(1, 2)
        channel_mask = mask.unsqueeze(1).to(values.dtype)
        values = values * channel_mask
        for block in self.blocks:
            values = block(values) * channel_mask

        count = channel_mask.sum(dim=2).clamp_min(1.0)
        mean_pool = values.sum(dim=2) / count
        masked = values.masked_fill(~mask.unsqueeze(1), torch.finfo(values.dtype).min)
        max_pool = masked.max(dim=2).values
        has_history = mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_history, max_pool, torch.zeros_like(max_pool))
        return self.pool_projection(torch.cat([mean_pool, max_pool], dim=-1))

