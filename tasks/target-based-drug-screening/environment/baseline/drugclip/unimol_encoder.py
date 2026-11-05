"""Checkpoint-compatible Uni-Mol v1 encoder for both DrugCLIP towers.

The module names and forward equations follow the official MIT-licensed
Uni-Mol v0.1 implementation. Only the active CLS representation path is kept;
masked-token, coordinate, and distance reconstruction heads are not needed for
virtual screening. Source: https://github.com/deepmodeling/Uni-Mol .
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _GaussianLayer(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.means = nn.Embedding(1, 128)
        self.stds = nn.Embedding(1, 128)
        self.mul = nn.Embedding(vocab_size * vocab_size, 1)
        self.bias = nn.Embedding(vocab_size * vocab_size, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.ones_(self.mul.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, distance: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        value = (self.mul(edge_type).type_as(distance) * distance.unsqueeze(-1)
                 + self.bias(edge_type).type_as(distance))
        mean = self.means.weight.float().view(128)
        std = self.stds.weight.float().view(128).abs() + 1e-5
        normalized = (value.float() - mean) / std
        result = torch.exp(-0.5 * normalized.square()) / (math.sqrt(2 * 3.14159) * std)
        return result.type_as(distance)


class _NonLinearHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(value)))


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(512, 1536)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, value: torch.Tensor, bias: torch.Tensor,
                attention_dropout: float, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = value.shape
        query, key, values = self.in_proj(value).chunk(3, dim=-1)
        query = query.reshape(batch, length, 64, 8).transpose(1, 2) * (8 ** -0.5)
        key = key.reshape(batch, length, 64, 8).transpose(1, 2)
        values = values.reshape(batch, length, 64, 8).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) + bias
        probabilities = F.dropout(F.softmax(logits, dim=-1),
                                  p=attention_dropout, training=training)
        attended = torch.matmul(probabilities, values)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, 512)
        return self.out_proj(attended), logits


class _EncoderLayer(nn.Module):
    def __init__(self, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()
        self.self_attn_layer_norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.final_layer_norm = nn.LayerNorm(512)
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def forward(self, value: torch.Tensor, bias: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        attention, bias = self.self_attn(
            self.self_attn_layer_norm(value), bias,
            self.attention_dropout, self.training,
        )
        value = value + F.dropout(attention, p=self.dropout, training=self.training)
        hidden = F.gelu(self.fc1(self.final_layer_norm(value)))
        value = value + F.dropout(self.fc2(hidden), p=self.dropout,
                                  training=self.training)
        return value, bias


class _Encoder(nn.Module):
    def __init__(self, dropout: float, attention_dropout: float,
                 gradient_checkpointing: bool) -> None:
        super().__init__()
        self.emb_layer_norm = nn.LayerNorm(512)
        self.final_layer_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList([
            _EncoderLayer(dropout, attention_dropout) for _ in range(15)
        ])
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, value: torch.Tensor, bias: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        value = F.dropout(self.emb_layer_norm(value), p=self.dropout,
                          training=self.training)
        value = value.masked_fill(padding_mask.unsqueeze(-1), 0)
        bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                value, bias = checkpoint(layer, value, bias, use_reentrant=False)
            else:
                value, bias = layer(value, bias)
        return self.final_layer_norm(value)[:, 0]


class UniMolEncoder(nn.Module):
    """The 15-layer/512-width active CLS path from Uni-Mol v0.1."""

    def __init__(self, vocab_size: int, dropout: float = 0.1,
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, 512, padding_idx=0)
        self.encoder = _Encoder(dropout, dropout, gradient_checkpointing)
        self.gbf_proj = _NonLinearHead()
        self.gbf = _GaussianLayer(vocab_size)
        self.apply(self._init_bert_params)

    @staticmethod
    def _init_bert_params(module: nn.Module) -> None:
        # Uni-Core applies this after constructing all Uni-Mol submodules.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"]
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("Uni-Mol token is outside its fixed dictionary")
        length = tokens.size(1)
        bias = self.gbf_proj(self.gbf(batch["distances"], batch["edge_types"]))
        bias = bias.permute(0, 3, 1, 2).contiguous()
        if bias.shape[-2:] != (length, length):
            raise ValueError("pair features disagree with token length")
        return self.encoder(self.embed_tokens(tokens), bias, tokens.eq(0))

    def load_official_checkpoint(self, path: Path, expected_sha256: str) -> dict:
        if self.vocab_size != 31:
            raise ValueError("the official v0.1 checkpoint is molecule-only")
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_sha256:
            raise ValueError("Uni-Mol molecule checkpoint SHA256 mismatch")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if set(raw) != {"model"} or not isinstance(raw["model"], dict):
            raise ValueError("unexpected Uni-Mol v0.1 checkpoint structure")
        excluded = ("lm_head.", "dist_head.", "pair2coord_proj.",
                    "encoder.final_head_layer_norm.")
        state = {key: tensor for key, tensor in raw["model"].items()
                 if not key.startswith(excluded)}
        if set(state) != set(self.state_dict()):
            missing = set(self.state_dict()) - set(state)
            unexpected = set(state) - set(self.state_dict())
            raise ValueError(f"Uni-Mol backbone keys differ: missing={sorted(missing)[:4]}, "
                             f"unexpected={sorted(unexpected)[:4]}")
        self.load_state_dict(state, strict=True)
        return {"sha256": actual, "loaded_tensors": len(state),
                "loaded_parameters": sum(value.numel() for value in state.values())}


class UniMolMoleculeEncoder(UniMolEncoder):
    def __init__(self, dropout: float = 0.1,
                 gradient_checkpointing: bool = True) -> None:
        super().__init__(31, dropout, gradient_checkpointing)
