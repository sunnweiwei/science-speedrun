"""Affordable multi-resolution baseline for the compact AlphaGenome task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import yaml

from .layers import ConvBlock, SequenceBlock


DENSE_TRACKS = {
    "atac": 167,
    "dnase": 305,
    "procap": 12,
    "cage": 546,
    "rna_seq": 667,
    "chip_tf": 1_617,
    "chip_histone": 1_116,
}
ALL_FAMILIES = (*DENSE_TRACKS, "splice_sites", "splice_usage", "junctions", "contact_maps")


@dataclass(frozen=True)
class ModelConfig:
    input_length: int = 1_048_576
    target_length: int = 196_608
    dense_bin_size: int = 128
    trunk_bin_size: int = 1_024
    contact_bin_size: int = 2_048
    stem_stride: int = 8
    encoder_width: int = 32
    channel_growth: int = 32
    num_downsampling_blocks: int = 7
    conv_kernel_size: int = 5
    dna_kernel_size: int = 15
    hidden_dim: int = 384
    num_sequence_blocks: int = 6
    num_attention_heads: int = 6
    attention_head_dim: int = 64
    mlp_expansion: float = 4.0
    dropout: float = 0.0
    head_hidden_size: int = 256
    junction_rank: int = 32
    splice_usage_tracks: int = 734
    junction_tracks: int = 734
    contact_tracks: int = 28
    gradient_checkpointing: bool = True
    enable_pairwise: bool = False
    pair_width: int = 32
    pair_downsample: int = 16
    pair_update_interval: int = 2

    def __post_init__(self) -> None:
        downsample = self.stem_stride * 2**self.num_downsampling_blocks
        if downsample != self.trunk_bin_size:
            raise ValueError("stem_stride * 2**num_downsampling_blocks must equal trunk_bin_size")
        for length in (self.input_length, self.target_length):
            if length % self.trunk_bin_size:
                raise ValueError("input and target lengths must divide by trunk_bin_size")
        if self.trunk_bin_size % self.dense_bin_size:
            raise ValueError("trunk_bin_size must divide by dense_bin_size")
        if self.input_length % self.contact_bin_size:
            raise ValueError("input_length must divide by contact_bin_size")
        if self.hidden_dim <= 0 or self.num_attention_heads <= 0:
            raise ValueError("hidden dimensions and attention heads must be positive")

    @property
    def trunk_bins(self) -> int:
        return self.input_length // self.trunk_bin_size

    @property
    def target_trunk_bins(self) -> int:
        return self.target_length // self.trunk_bin_size

    @property
    def dense_bins(self) -> int:
        return self.target_length // self.dense_bin_size

    @property
    def dense_upsample(self) -> int:
        return self.trunk_bin_size // self.dense_bin_size

    @property
    def contact_bins(self) -> int:
        return self.input_length // self.contact_bin_size

    @property
    def encoder_output_width(self) -> int:
        return self.encoder_width + self.channel_growth * self.num_downsampling_blocks

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        values: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        known = {field.name for field in fields(cls)}
        unknown = sorted(values.keys() - known)
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {unknown}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SequenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.stem = nn.Conv1d(
            4,
            config.encoder_width,
            config.dna_kernel_size,
            stride=config.stem_stride,
            padding=config.dna_kernel_size // 2,
        )
        blocks = []
        width = config.encoder_width
        for _ in range(config.num_downsampling_blocks):
            output_width = width + config.channel_growth
            blocks.append(ConvBlock(width, output_width, config.conv_kernel_size))
            width = output_width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, dna: Tensor) -> Tensor:
        x = F.gelu(self.stem(dna.transpose(1, 2)), approximate="tanh")
        for block in self.blocks:
            x = F.max_pool1d(block(x), kernel_size=2, stride=2)
        return x.transpose(1, 2)


class CompactModel(nn.Module):
    """One shared 1-kb trunk with representation-specific output heads."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = SequenceEncoder(self.config)
        self.projection = nn.Linear(self.config.encoder_output_width, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            SequenceBlock(self.config) for _ in range(self.config.num_sequence_blocks)
        )
        self.final_norm = nn.RMSNorm(self.config.hidden_dim)
        self.dense_decoder = nn.Sequential(
            nn.RMSNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.head_hidden_size),
            nn.GELU(approximate="tanh"),
        )
        self.dense_heads = nn.ModuleDict(
            {
                family: nn.Linear(self.config.head_hidden_size, tracks)
                for family, tracks in DENSE_TRACKS.items()
            }
        )
        self.splice_site_head = nn.Linear(
            self.config.head_hidden_size, self.config.dense_bin_size * 4
        )
        self.splice_usage_head = nn.Linear(
            self.config.head_hidden_size, self.config.splice_usage_tracks
        )
        rank = self.config.junction_rank
        self.junction_source = nn.Linear(self.config.head_hidden_size, rank)
        self.junction_destination = nn.Linear(self.config.head_hidden_size, rank)
        self.junction_track = nn.Embedding(self.config.junction_tracks, rank)
        if self.config.junction_tracks % 2:
            raise ValueError("junction tracks must split evenly into strand halves")
        self.register_buffer(
            "junction_track_half",
            torch.arange(self.config.junction_tracks) // (self.config.junction_tracks // 2),
            persistent=False,
        )
        self.contact_q = nn.Linear(self.config.hidden_dim, self.config.contact_tracks)
        self.contact_k = nn.Linear(self.config.hidden_dim, self.config.contact_tracks)
        row, column = torch.triu_indices(
            self.config.contact_bins, self.config.contact_bins
        )
        self.register_buffer("contact_row", row, persistent=False)
        self.register_buffer("contact_column", column, persistent=False)

    def set_junction_track_halves(self, halves: Tensor) -> None:
        halves = torch.as_tensor(halves, dtype=torch.long, device=self.junction_track_half.device)
        if halves.shape != (self.config.junction_tracks,) or not torch.isin(
            halves, torch.tensor([0, 1], device=halves.device)
        ).all():
            raise ValueError("junction track halves must be a 734-element 0/1 mapping")
        self.junction_track_half.copy_(halves)

    def _trunk(self, dna: Tensor) -> Tensor:
        if dna.ndim != 3 or dna.shape[1:] != (self.config.input_length, 4):
            raise ValueError(
                f"Expected DNA [B,{self.config.input_length},4], got {tuple(dna.shape)}"
            )
        x = self.projection(self.encoder(dna))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(block, x, None, use_reentrant=False)
            else:
                x = block(x, None)
        return self.final_norm(x)

    def _dense_features(self, trunk: Tensor) -> Tensor:
        start = (self.config.trunk_bins - self.config.target_trunk_bins) // 2
        target = trunk[:, start : start + self.config.target_trunk_bins]
        target = target.repeat_interleave(self.config.dense_upsample, dim=1)
        return self.dense_decoder(target)

    def _all_dense_features(self, trunk: Tensor) -> Tensor:
        return self.dense_decoder(
            trunk.repeat_interleave(self.config.dense_upsample, dim=1)
        )

    def _splice_site_logits(self, trunk: Tensor) -> Tensor:
        logits = self.splice_site_head(self._all_dense_features(trunk))
        return logits.view(len(trunk), self.config.input_length, 4).float()

    @staticmethod
    def select_junction_positions(site_logits: Tensor, limit: int = 512) -> Tensor:
        """Select each site type from model scores alone, then sort by coordinate.

        This follows the four independent site classes in Google DeepMind's
        AlphaGenome splicing implementation (pinned in docs/v3-author/PROTOCOL.md).  Padding is
        -1 when an input is shorter than ``limit``.
        """
        count = min(limit, site_logits.shape[1])
        selected = []
        for site_type in range(4):
            coordinates = torch.topk(site_logits[..., site_type], count, dim=1).indices
            coordinates = coordinates.sort(dim=1).values
            if count < limit:
                coordinates = F.pad(coordinates, (0, limit - count), value=-1)
            selected.append(coordinates)
        return torch.stack(selected, dim=1)

    @staticmethod
    def _gather_bins(features: Tensor, positions: Tensor, bin_size: int) -> Tensor:
        indices = torch.div(positions, bin_size, rounding_mode="floor")
        indices = indices.clamp(0, features.shape[1] - 1)
        return torch.gather(
            features,
            1,
            indices[..., None].expand(-1, -1, features.shape[-1]),
        )

    def _junction_predictions(
        self, trunk: Tensor, site_positions: Tensor, queries: Tensor
    ) -> Tensor:
        row, column, track = queries.unbind(dim=-1)
        half = self.junction_track_half[track]
        batch = torch.arange(len(trunk), device=trunk.device)[:, None]
        donor = site_positions[batch, 2 * half, row]
        acceptor = site_positions[batch, 2 * half + 1, column]
        source = self.dense_decoder(self._gather_bins(
            trunk, donor.clamp(0, self.config.input_length - 1), self.config.trunk_bin_size
        ))
        destination = self.dense_decoder(self._gather_bins(
            trunk, acceptor.clamp(0, self.config.input_length - 1), self.config.trunk_bin_size
        ))
        interaction = self.junction_source(source) * self.junction_destination(destination)
        track_embedding = self.junction_track(track)
        return (interaction * track_embedding).sum(dim=-1) / math.sqrt(
            self.config.junction_rank
        )

    def junction_block(
        self,
        trunk: Tensor,
        site_positions: Tensor,
        row_start: int,
        row_end: int,
        column_start: int,
        column_end: int,
        track_start: int,
        track_end: int,
        site_factors: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Compute a bounded block of the full predicted-position junction cube."""
        tracks = torch.arange(track_start, track_end, device=trunk.device)
        halves = self.junction_track_half[tracks]
        if site_factors is None:
            site_factors = self.junction_site_factors(trunk, site_positions)
        source_all, destination_all = site_factors
        source = source_all[:, halves, row_start:row_end]
        destination = destination_all[:, halves, column_start:column_end]
        embedding = self.junction_track(tracks)[None, :, None, None, :]
        values = (
            source[:, :, :, None, :] * destination[:, :, None, :, :] * embedding
        ).sum(dim=-1) / math.sqrt(self.config.junction_rank)
        return values.permute(0, 2, 3, 1).float()

    def junction_site_factors(
        self, trunk: Tensor, site_positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Precompute the two strand-specific factors reused by all cube blocks."""
        donor = torch.stack((site_positions[:, 0], site_positions[:, 2]), dim=1).clamp_min(0)
        acceptor = torch.stack((site_positions[:, 1], site_positions[:, 3]), dim=1).clamp_min(0)
        expanded = trunk[:, None].expand(-1, 2, -1, -1).reshape(
            -1, trunk.shape[1], trunk.shape[2]
        )
        donor_features = self.dense_decoder(self._gather_bins(
            expanded, donor.reshape(-1, donor.shape[-1]), self.config.trunk_bin_size,
        )).reshape(len(trunk), 2, donor.shape[-1], -1)
        acceptor_features = self.dense_decoder(self._gather_bins(
            expanded, acceptor.reshape(-1, acceptor.shape[-1]), self.config.trunk_bin_size,
        )).reshape(len(trunk), 2, acceptor.shape[-1], -1)
        return self.junction_source(donor_features), self.junction_destination(acceptor_features)

    def forward(
        self,
        dna: Tensor,
        *,
        families: Iterable[str] | None = None,
        usage_positions: Tensor | None = None,
        junction_site_positions: Tensor | None = None,
        junction_queries: Tensor | None = None,
    ) -> dict[str, Tensor]:
        requested = set(families or ALL_FAMILIES)
        unknown = requested - set(ALL_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown output families: {sorted(unknown)}")
        trunk = self._trunk(dna)
        needs_dense = bool(
            requested.intersection((*DENSE_TRACKS, "splice_sites", "splice_usage", "junctions"))
        )
        features = self._dense_features(trunk) if needs_dense else None
        outputs: dict[str, Tensor] = {}
        for family in DENSE_TRACKS:
            if family in requested:
                outputs[family] = self.dense_heads[family](features).float()
        if "splice_sites" in requested:
            all_logits = self._splice_site_logits(trunk)
            start = (self.config.input_length - self.config.target_length) // 2
            outputs["splice_sites"] = all_logits[:, start:start + self.config.target_length]
        if "splice_usage" in requested:
            if usage_positions is None:
                raise ValueError("splice_usage requires usage_positions")
            gathered = self._gather_bins(
                features, usage_positions, self.config.dense_bin_size
            )
            outputs["splice_usage"] = self.splice_usage_head(gathered).float()
        if "junctions" in requested:
            if junction_site_positions is None or junction_queries is None:
                raise ValueError("junctions require site positions and query triples")
            outputs["junctions"] = self._junction_predictions(
                trunk, junction_site_positions, junction_queries
            ).float()
        if "contact_maps" in requested:
            contact = trunk.reshape(
                len(dna), self.config.contact_bins, -1, self.config.hidden_dim
            ).mean(dim=2)
            q, k = self.contact_q(contact), self.contact_k(contact)
            row, column = self.contact_row, self.contact_column
            outputs["contact_maps"] = (
                0.5
                * (
                    q[:, row] * k[:, column]
                    + q[:, column] * k[:, row]
                )
            ).float()
        return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
