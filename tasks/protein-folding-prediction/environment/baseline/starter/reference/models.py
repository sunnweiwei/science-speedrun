"""Compact reference folding models inspired by ESMFold2 and AF3-style trunks."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True)
class ReferenceFoldingConfig:
    architecture: str = "pairformer_lite"
    d_single: int = 448
    d_pair: int = 128
    n_trunk_blocks: int = 18
    n_structure_blocks: int = 4
    n_heads: int = 8
    triangle_hidden: int = 64
    transition_factor: int = 4
    dropout: float = 0.0
    distogram_bins: int = 64
    max_relative_index: int = 32

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ReferenceFoldingConfig":
        unknown = set(values) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown model config fields: {sorted(unknown)}")
        return cls(**values)


class Transition(nn.Module):
    def __init__(self, dim: int, factor: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, dim * factor)
        self.linear_out = nn.Linear(dim * factor, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.linear_out(F.silu(self.linear_in(self.norm(value))))
        return value + self.dropout(update)


class PairBiasedSelfAttention(nn.Module):
    def __init__(
        self, d_single: int, d_pair: int, n_heads: int, dropout: float, pair_bias: bool
    ) -> None:
        super().__init__()
        if d_single % n_heads:
            raise ValueError("d_single must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_single // n_heads
        self.norm = nn.LayerNorm(d_single)
        self.qkv = nn.Linear(d_single, 3 * d_single, bias=False)
        self.output = nn.Linear(d_single, d_single, bias=False)
        self.pair_bias = nn.Linear(d_pair, n_heads, bias=False) if pair_bias else None
        self.dropout = dropout

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, channels = single.shape
        qkv = self.qkv(self.norm(single)).view(
            batch, length, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.pair_bias is not None:
            scores = scores + self.pair_bias(pair).permute(0, 3, 1, 2)
        key_mask = token_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        update = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, length, channels
        )
        update = self.output(update) * token_mask.unsqueeze(-1)
        return single + update


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.left = nn.Linear(d_pair, hidden)
        self.right = nn.Linear(d_pair, hidden)
        self.output = nn.Linear(hidden, d_pair)
        self.gate = nn.Linear(d_pair, d_pair)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(pair)
        left = self.left(normalized) * pair_mask.unsqueeze(-1)
        right = self.right(normalized) * pair_mask.unsqueeze(-1)
        update = torch.einsum("bikd,bjkd->bijd", left, right)
        update = update / math.sqrt(max(1, pair.shape[1]))
        update = self.output(update) * torch.sigmoid(self.gate(normalized))
        update = update * pair_mask.unsqueeze(-1)
        return pair + self.dropout(update)


class OuterProductUpdate(nn.Module):
    def __init__(self, d_single: int, d_pair: int, hidden: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_single)
        self.left = nn.Linear(d_single, hidden)
        self.right = nn.Linear(d_single, hidden)
        self.output = nn.Linear(hidden, d_pair)

    def forward(self, single: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        value = self.norm(single)
        update = self.left(value).unsqueeze(2) * self.right(value).unsqueeze(1)
        return self.output(update) * pair_mask.unsqueeze(-1)


class TrunkBlock(nn.Module):
    def __init__(self, config: ReferenceFoldingConfig, pair_bias: bool) -> None:
        super().__init__()
        self.single_attention = PairBiasedSelfAttention(
            config.d_single,
            config.d_pair,
            config.n_heads,
            config.dropout,
            pair_bias=pair_bias,
        )
        self.single_transition = Transition(
            config.d_single, config.transition_factor, config.dropout
        )
        self.outer_product = OuterProductUpdate(config.d_single, config.d_pair)
        self.triangle = TriangleMultiplication(
            config.d_pair, config.triangle_hidden, config.dropout
        )
        self.pair_transition = Transition(
            config.d_pair, config.transition_factor, config.dropout
        )

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single = self.single_attention(single, pair, token_mask)
        single = self.single_transition(single) * token_mask.unsqueeze(-1)
        pair = pair + self.outer_product(single, pair_mask)
        pair = self.triangle(pair, pair_mask)
        pair = self.pair_transition(pair) * pair_mask.unsqueeze(-1)
        return single, pair


class EquivariantCoordinateBlock(nn.Module):
    def __init__(self, d_pair: int, n_rbf: int = 24) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(d_pair)
        self.message = nn.Sequential(
            nn.Linear(d_pair + n_rbf, d_pair),
            nn.SiLU(),
            nn.Linear(d_pair, d_pair),
            nn.SiLU(),
            nn.Linear(d_pair, 1),
        )
        nn.init.zeros_(self.message[-1].weight)
        nn.init.zeros_(self.message[-1].bias)
        self.register_buffer("rbf_centers", torch.linspace(0.0, 4.0, n_rbf))

    def forward(
        self, coords: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor
    ) -> torch.Tensor:
        displacement = coords.unsqueeze(1) - coords.unsqueeze(2)
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1).to(coords.dtype)
        rbf = torch.exp(
            -4.0 * (distance.unsqueeze(-1) - self.rbf_centers.to(distance.dtype)) ** 2
        )
        weight = torch.tanh(self.message(torch.cat((self.pair_norm(pair), rbf), dim=-1)))
        weight = weight * pair_mask.unsqueeze(-1)
        normalizer = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(coords.dtype)
        update = (weight * displacement).sum(dim=2) / normalizer
        return coords + update


class ReferenceFoldingModel(nn.Module):
    def __init__(self, config: ReferenceFoldingConfig) -> None:
        super().__init__()
        if config.architecture not in {"esmfold_lite", "pairformer_lite"}:
            raise ValueError(
                "implemented architectures are esmfold_lite and pairformer_lite; "
                f"got {config.architecture!r}"
            )
        self.config = config
        self.lm_projection = nn.Sequential(
            nn.LayerNorm(2560), nn.Linear(2560, config.d_single, bias=False)
        )
        self.molecule_embedding = nn.Embedding(8, config.d_single)
        self.residue_embedding = nn.Embedding(512, config.d_single)
        self.atom_embedding = nn.Embedding(128, config.d_single)
        self.task_embedding = nn.Embedding(11, config.d_single)
        self.lm_presence_embedding = nn.Embedding(2, config.d_single)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, config.d_single),
            nn.SiLU(),
            nn.Linear(config.d_single, config.d_single),
        )
        self.left_pair = nn.Linear(config.d_single, config.d_pair, bias=False)
        self.right_pair = nn.Linear(config.d_single, config.d_pair, bias=False)
        relative_bins = 2 * config.max_relative_index + 2
        self.relative_embedding = nn.Embedding(relative_bins, config.d_pair)
        self.same_chain_embedding = nn.Embedding(2, config.d_pair)
        pair_bias = config.architecture == "pairformer_lite"
        self.trunk = nn.ModuleList(
            TrunkBlock(config, pair_bias=pair_bias)
            for _ in range(config.n_trunk_blocks)
        )
        self.structure = nn.ModuleList(
            EquivariantCoordinateBlock(config.d_pair)
            for _ in range(config.n_structure_blocks)
        )
        self.distogram_norm = nn.LayerNorm(config.d_pair)
        self.distogram_head = nn.Linear(config.d_pair, config.distogram_bins)

    def _initial_pair(
        self,
        single: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
    ) -> torch.Tensor:
        pair = self.left_pair(single).unsqueeze(2) + self.right_pair(single).unsqueeze(1)
        relative = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)
        clipped = relative.clamp(
            -self.config.max_relative_index, self.config.max_relative_index
        ) + self.config.max_relative_index
        clipped = torch.where(
            same_chain,
            clipped,
            torch.full_like(clipped, 2 * self.config.max_relative_index + 1),
        )
        pair = pair + self.relative_embedding(clipped)
        pair = pair + self.same_chain_embedding(same_chain.long())
        return pair

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_mask = batch["token_mask"]
        pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        molecule_type = batch["molecule_type"].clamp(0, 7)
        log_sigma = torch.log(batch["sigma"].clamp_min(1e-4)).unsqueeze(-1)
        single = (
            self.lm_projection(batch["single"])
            + self.molecule_embedding(molecule_type)
            + self.residue_embedding(batch["residue_type"])
            + self.atom_embedding(batch["atom_type"])
            + self.task_embedding(batch["task_id"]).unsqueeze(1)
            + self.lm_presence_embedding(batch["lm_mask"].long())
            + self.sigma_mlp(log_sigma).unsqueeze(1)
        )
        single = single * token_mask.unsqueeze(-1)
        pair = self._initial_pair(single, batch["residue_index"], batch["chain_id"])
        pair = pair * pair_mask.unsqueeze(-1)
        for block in self.trunk:
            single, pair = block(single, pair, token_mask, pair_mask)
        coords = batch["noisy_coords"]
        for block in self.structure:
            coords = block(coords, pair, pair_mask)
            coords = coords * token_mask.unsqueeze(-1)
        logits = self.distogram_head(self.distogram_norm(pair))
        return {"pred_coords": coords, "distogram_logits": logits, "single": single}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: ReferenceFoldingConfig | dict[str, Any]) -> ReferenceFoldingModel:
    if isinstance(config, dict):
        config = ReferenceFoldingConfig.from_dict(config)
    return ReferenceFoldingModel(config)
