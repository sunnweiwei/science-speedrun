"""Shared neural-network layers for the single compact AlphaGenome model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().square().mean(dim=1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * self.weight[None, :, None].to(x.dtype)


class StandardizedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size, padding="same", bias=True)
        self.scale = nn.Parameter(torch.ones(out_channels))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(in_channels * kernel_size))
        nn.init.trunc_normal_(self.bias, std=1e-4)

    def forward(self, x: Tensor) -> Tensor:
        centered = self.weight - self.weight.mean(dim=(1, 2), keepdim=True)
        variance = centered.float().square().mean(dim=(1, 2), keepdim=True)
        fan_in = self.weight.shape[1] * self.weight.shape[2]
        multiplier = self.scale[:, None, None] * torch.rsqrt(
            (fan_in * variance).clamp_min(1e-4)
        ).to(self.weight.dtype)
        return F.conv1d(x, centered * multiplier, self.bias, padding="same")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.norm = ChannelRMSNorm(in_channels)
        self.conv = StandardizedConv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.gelu(self.norm(x), approximate="tanh"))


def _apply_rope(x: Tensor) -> Tensor:
    rotary_dim = x.shape[-1] - x.shape[-1] % 2
    if rotary_dim == 0:
        return x
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (
        10_000 ** (
            torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    angles = torch.outer(positions, inverse_frequency)
    cos = angles.cos()[None, None, :, :].to(x.dtype)
    sin = angles.sin()[None, None, :, :].to(x.dtype)
    even, odd = x[..., :rotary_dim:2], x[..., 1:rotary_dim:2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return torch.cat((rotated.flatten(-2), x[..., rotary_dim:]), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        inner_dim = self.num_heads * self.head_dim
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.qkv = nn.Linear(config.hidden_dim, 3 * inner_dim, bias=False)
        self.output = nn.Linear(inner_dim, config.hidden_dim)
        self.dropout = config.dropout
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor, attention_bias: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm(x)).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            _apply_rope(q.transpose(1, 2)),
            _apply_rope(k.transpose(1, 2)),
            value.transpose(1, 2),
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, -1))


class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        expanded = int(config.hidden_dim * config.mlp_expansion)
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.input = nn.Linear(config.hidden_dim, expanded)
        self.output = nn.Linear(expanded, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.trunc_normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(
            self.output(F.gelu(self.input(self.norm(x)), approximate="tanh"))
        )


class SequenceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLPBlock(config)

    def forward(self, x: Tensor, attention_bias: Tensor | None = None) -> Tensor:
        x = x + self.attention(x, attention_bias)
        return x + self.mlp(x)
