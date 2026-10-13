"""Starter model: a small decoder-only patched transformer with quantile heads (TimesFM-shaped, from scratch).

Input: a context of up to `max_context` observations, standardized per series
by the mean and standard deviation of its finite values (REVIN-style), split
into non-overlapping patches of `patch` observations. Each patch is embedded
with its missing-value mask; a causal transformer maps every position to a
forecast of the next `horizon` observations: nine quantile levels plus a
mean. Longer horizons are produced autoregressively by feeding the median
back as context. About 6.4M parameters at the default size.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PatchForecaster(nn.Module):
    def __init__(self, patch=32, horizon=128, d_model=256, layers=6, heads=4, max_context=2048, dropout=0.0):
        super().__init__()
        assert max_context % patch == 0
        self.patch, self.horizon, self.max_context = patch, horizon, max_context
        self.n_out = len(QUANTILES) + 1
        self.embed = nn.Sequential(nn.Linear(2 * patch, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_context // patch, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(d_model, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, horizon * self.n_out))

    @staticmethod
    def standardize(ctx: torch.Tensor):
        """ctx (B, T) with NaN for missing -> (z (B, T) with 0 at missing, mask (B, T), mu (B, 1), sigma (B, 1))."""
        finite = torch.isfinite(ctx)
        x = torch.where(finite, ctx, torch.zeros_like(ctx))
        n = finite.sum(1, keepdim=True).clamp(min=1)
        mu = x.sum(1, keepdim=True) / n
        var = ((x - mu) ** 2 * finite).sum(1, keepdim=True) / n
        # scale floor relative to the level, so near-constant contexts do not blow up the standardized future
        sigma = torch.maximum(torch.sqrt(var + 1e-6), 1e-2 * mu.abs() + 1e-3)
        z = torch.where(finite, (x - mu) / sigma, torch.zeros_like(x))
        return z, finite.float(), mu, sigma

    def forward_tokens(self, z: torch.Tensor, mask: torch.Tensor):
        """z, mask (B, T) with T a multiple of patch -> per-position outputs (B, T/patch, n_out, horizon) in standardized units."""
        b, t = z.shape
        p = t // self.patch
        tokens = torch.cat([z.view(b, p, self.patch), mask.view(b, p, self.patch)], dim=-1)
        h = self.embed(tokens) + self.pos[:, -p:]
        causal = torch.triu(torch.ones(p, p, dtype=torch.bool, device=z.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, causal)
        out = self.head(self.norm(h)).view(b, p, self.n_out, self.horizon)
        return out

    @torch.no_grad()
    def predict(self, ctx: torch.Tensor, steps: int):
        """ctx (B, T) float with NaN padding/missing -> (mean (B, steps), quantiles (B, 9, steps)) in original units."""
        ctx = ctx[:, -self.max_context:]
        # always left-pad to max_context so that every instance sees the same positional slots regardless of how it is batched
        pad = self.max_context - ctx.shape[1]
        if pad:
            ctx = F.pad(ctx, (pad, 0), value=float("nan"))
        z, mask, mu, sigma = self.standardize(ctx)
        means, quants = [], []
        produced = 0
        while produced < steps:
            out = self.forward_tokens(z[:, -self.max_context:], mask[:, -self.max_context:])[:, -1]  # (B, n_out, horizon)
            q = out[:, :len(QUANTILES)]
            q, _ = torch.sort(q, dim=1)
            m = out[:, len(QUANTILES)]
            take = min(self.horizon, steps - produced)
            means.append(m[:, :take]); quants.append(q[:, :, :take]); produced += take
            if produced < steps:
                z = torch.cat([z, q[:, len(QUANTILES) // 2]], dim=1)
                mask = torch.cat([mask, torch.ones_like(q[:, 0])], dim=1)
        mean = torch.cat(means, 1) * sigma + mu
        quant = torch.cat(quants, 2) * sigma[:, :, None] + mu[:, :, None]
        return mean, quant


def quantile_loss(pred_q: torch.Tensor, target: torch.Tensor, levels=QUANTILES):
    """pred_q (..., 9, H), target (..., H) with NaN = missing -> mean pinball loss over finite targets."""
    finite = torch.isfinite(target)
    t = torch.where(finite, target, torch.zeros_like(target))[..., None, :]
    lv = torch.tensor(levels, device=pred_q.device, dtype=pred_q.dtype).view(*([1] * (pred_q.dim() - 2)), -1, 1)
    err = t - pred_q
    loss = torch.maximum(lv * err, (lv - 1) * err)
    w = finite[..., None, :].expand_as(loss).float()
    return (loss * w).sum() / w.sum().clamp(min=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
