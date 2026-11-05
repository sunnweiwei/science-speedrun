"""Paper-scale dual Uni-Mol reference and historical scratch baselines.

PaperDrugCLIP loads the official Uni-Mol molecule backbone and learns a
matching Uni-Mol pocket tower through ProFSA and BioLiP stages.
CompactDrugCLIP remains for earlier from-scratch ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from unimol_encoder import UniMolEncoder


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    encoder_width: int = 512
    encoder_layers: int = 15
    attention_heads: int = 64
    ffn_multiplier: float = 4.0
    embedding_dim: int = 128
    distance_basis: int = 128
    dropout: float = 0.1
    max_molecule_atoms: int = 256
    max_pocket_atoms: int = 256
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include special and atom tokens")
        if self.encoder_width % self.attention_heads:
            raise ValueError("encoder_width must be divisible by attention_heads")
        if min(self.encoder_width, self.encoder_layers, self.attention_heads,
               self.embedding_dim, self.distance_basis, self.max_molecule_atoms,
               self.max_pocket_atoms) < 1:
            raise ValueError("model dimensions must be positive")
        if self.ffn_multiplier < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid ffn_multiplier or dropout")

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # Total parameter sizes include both independent towers.
    "tiny": {"encoder_width": 128, "encoder_layers": 2,
             "attention_heads": 4, "distance_basis": 32},
    # Historical compact DrugCLIP-style scratch reference, not the original.
    "drugclip-50m": {"encoder_width": 448, "encoder_layers": 10,
                     "attention_heads": 8, "distance_basis": 128},
    # Active architecture from the released DrugCLIP paper code. The source
    # also declares auxiliary heads that its contrastive forward path does not
    # use; this preset reproduces the two encoders and projection heads that
    # determine embeddings and screening scores.
    "drugclip-original": {"encoder_width": 512, "encoder_layers": 15,
                          "attention_heads": 64, "distance_basis": 128},
}


def preset_config(name: str, vocab_size: int, **overrides) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = ModelConfig(vocab_size=vocab_size, **values)
    config.validate()
    return config


class GaussianDistanceBias(nn.Module):
    """Uni-Mol-style edge-conditioned Gaussian radial basis attention bias."""

    def __init__(self, vocab_size: int, basis_size: int, heads: int) -> None:
        super().__init__()
        edge_types = vocab_size * vocab_size
        self.edge_scale = nn.Embedding(edge_types, 1)
        self.edge_shift = nn.Embedding(edge_types, 1)
        means = torch.linspace(0.0, 20.0, basis_size)
        self.means = nn.Parameter(means)
        self.log_widths = nn.Parameter(torch.zeros(basis_size))
        self.projection = nn.Sequential(
            nn.Linear(basis_size, basis_size), nn.GELU(), nn.Linear(basis_size, heads)
        )
        nn.init.ones_(self.edge_scale.weight)
        nn.init.zeros_(self.edge_shift.weight)

    def forward(self, distances: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        scaled = distances.unsqueeze(-1) * self.edge_scale(edge_types)
        scaled = scaled + self.edge_shift(edge_types)
        widths = F.softplus(self.log_widths).clamp_min(1e-3)
        basis = torch.exp(-0.5 * ((scaled - self.means) / widths).square())
        return self.projection(basis).permute(0, 3, 1, 2).contiguous()


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, values: torch.Tensor, pair_bias: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = values.shape
        qkv = self.qkv(values).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key_mask = padding_mask[:, None, None, :]
        attention_bias = pair_bias.masked_fill(key_mask, float("-inf"))
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class EncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiplier: float,
                 dropout: float) -> None:
        super().__init__()
        hidden = int(round(width * ffn_multiplier))
        self.attention_norm = nn.LayerNorm(width)
        self.attention = PairBiasedSelfAttention(width, heads, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, pair_bias: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        values = values + self.dropout(
            self.attention(self.attention_norm(values), pair_bias, padding_mask)
        )
        values = values + self.dropout(self.ffn(self.ffn_norm(values)))
        return values


class StructureEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.padding_index = 0
        self.tokens = nn.Embedding(config.vocab_size, config.encoder_width,
                                   padding_idx=self.padding_index)
        self.distance_bias = GaussianDistanceBias(
            config.vocab_size, config.distance_basis, config.attention_heads
        )
        self.layers = nn.ModuleList([
            EncoderLayer(config.encoder_width, config.attention_heads,
                         config.ffn_multiplier, config.dropout)
            for _ in range(config.encoder_layers)
        ])
        self.final_norm = nn.LayerNorm(config.encoder_width)
        self.use_checkpointing = config.gradient_checkpointing

    def forward(self, tokens: torch.Tensor, distances: torch.Tensor,
                edge_types: torch.Tensor) -> torch.Tensor:
        padding_mask = tokens.eq(self.padding_index)
        values = self.tokens(tokens)
        pair_bias = self.distance_bias(distances, edge_types)
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                values = checkpoint(layer, values, pair_bias, padding_mask,
                                    use_reentrant=False)
            else:
                values = layer(values, pair_bias, padding_mask)
        return self.final_norm(values)[:, 0]


class ProjectionHead(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(source, source), nn.ReLU(), nn.Linear(source, target),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(values), dim=-1)


class CompactDrugCLIP(nn.Module):
    """Independent molecule/pocket encoders in one normalized embedding space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.molecule_encoder = StructureEncoder(config)
        self.pocket_encoder = StructureEncoder(config)
        self.molecule_projection = ProjectionHead(config.encoder_width,
                                                  config.embedding_dim)
        self.pocket_projection = ProjectionHead(config.encoder_width,
                                                config.embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(14.0)))

    def encode_molecule(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        representation = self.molecule_encoder(
            batch["tokens"], batch["distances"], batch["edge_types"]
        )
        return self.molecule_projection(representation)

    def encode_pocket(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        representation = self.pocket_encoder(
            batch["tokens"], batch["distances"], batch["edge_types"]
        )
        return self.pocket_projection(representation)

    def forward(self, molecule: dict[str, torch.Tensor],
                pocket: dict[str, torch.Tensor]) -> torch.Tensor:
        molecule_embedding = self.encode_molecule(molecule)
        pocket_embedding = self.encode_pocket(pocket)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * pocket_embedding @ molecule_embedding.T

    @staticmethod
    def contrastive_loss(logits: torch.Tensor,
                         molecule_ids: list[str] | None = None,
                         pocket_ids: list[str] | None = None) -> torch.Tensor:
        """Symmetric in-batch InfoNCE with DrugCLIP duplicate masking."""
        if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
            raise ValueError("contrastive logits must be a square matrix")
        size = logits.shape[0]
        if size < 2:
            raise ValueError("contrastive batches need at least two examples")
        masked = logits
        if molecule_ids is not None and pocket_ids is not None:
            if len(molecule_ids) != size or len(pocket_ids) != size:
                raise ValueError("ID count does not match contrastive batch")
            duplicate = torch.zeros_like(logits, dtype=torch.bool)
            for row in range(size):
                for column in range(size):
                    if row != column and (molecule_ids[row] == molecule_ids[column]
                                          or pocket_ids[row] == pocket_ids[column]):
                        duplicate[row, column] = True
            masked = logits.masked_fill(duplicate, -torch.finfo(logits.dtype).max)
        target = torch.arange(size, device=logits.device)
        return 0.5 * (F.cross_entropy(masked.float(), target)
                      + F.cross_entropy(masked.float().T, target))


@dataclass(frozen=True)
class PaperConfig:
    molecule_vocab_size: int = 31
    pocket_vocab_size: int = 10
    encoder_layers: int = 15
    encoder_width: int = 512
    ffn_width: int = 2048
    attention_heads: int = 64
    embedding_dim: int = 128
    dropout: float = 0.1
    max_molecule_atoms: int = 256
    max_pocket_atoms: int = 256
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if (self.molecule_vocab_size, self.pocket_vocab_size,
                self.encoder_layers, self.encoder_width, self.ffn_width,
                self.attention_heads, self.embedding_dim) != (31, 10, 15, 512, 2048, 64, 128):
            raise ValueError("paper reference requires both fixed Uni-Mol architectures")
        if min(self.max_molecule_atoms, self.max_pocket_atoms) < 1:
            raise ValueError("atom limits must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


class PaperDrugCLIP(nn.Module):
    """Paper-scale dual Uni-Mol model with fixed scale 14."""

    def __init__(self, config: PaperConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.molecule_encoder = UniMolEncoder(
            config.molecule_vocab_size, config.dropout,
            config.gradient_checkpointing,
        )
        self.pocket_encoder = UniMolEncoder(
            config.pocket_vocab_size, config.dropout,
            config.gradient_checkpointing,
        )
        self.molecule_projection = ProjectionHead(512, 128)
        self.pocket_projection = ProjectionHead(512, 128)
        self.register_buffer("logit_scale", torch.tensor(math.log(14.0)))

    def set_stage(self, stage: str) -> None:
        if stage not in {"profsa", "finetune"}:
            raise ValueError(f"unknown training stage {stage}")
        for parameter in self.molecule_encoder.parameters():
            parameter.requires_grad_(stage == "finetune")
        for parameter in self.pocket_encoder.parameters():
            parameter.requires_grad_(True)
        for module in (self.molecule_projection, self.pocket_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.stage = stage

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "stage", None) == "profsa":
            self.molecule_encoder.eval()
        return self

    def encode_molecule(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if getattr(self, "stage", None) == "profsa":
            with torch.no_grad():
                representation = self.molecule_encoder(batch)
        else:
            representation = self.molecule_encoder(batch)
        return self.molecule_projection(representation)

    def encode_pocket(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.pocket_projection(self.pocket_encoder(batch))

    def forward(self, molecule: dict[str, torch.Tensor],
                pocket: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logit_scale.exp().detach() * (
            self.encode_pocket(pocket) @ self.encode_molecule(molecule).T
        )

    @staticmethod
    def contrastive_loss(logits: torch.Tensor, molecule_ids: list[str],
                         pocket_ids: list[str]) -> torch.Tensor:
        masked = author_masked_logits(logits.float(), molecule_ids, pocket_ids)
        target = torch.arange(masked.shape[0], device=masked.device)
        return 0.5 * (F.cross_entropy(masked, target)
                      + F.cross_entropy(masked.T, target))


def author_masked_logits(logits: torch.Tensor, molecule_ids: list[str],
                         pocket_ids: list[str]) -> torch.Tensor:
    """The author's off-diagonal molecule/pocket duplicate penalty."""
    size = logits.shape[0]
    if logits.ndim != 2 or logits.shape != (size, size):
        raise ValueError("retrieval logits must be square")
    if len(molecule_ids) != size or len(pocket_ids) != size:
        raise ValueError("retrieval IDs disagree with batch size")
    penalty = torch.zeros((size, size), device=logits.device, dtype=logits.dtype)
    for row in range(size):
        for col in range(size):
            if row != col:
                penalty[row, col] = (int(molecule_ids[row] == molecule_ids[col])
                                     + int(pocket_ids[row] == pocket_ids[col]))
    return logits - 1e6 * penalty



def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
