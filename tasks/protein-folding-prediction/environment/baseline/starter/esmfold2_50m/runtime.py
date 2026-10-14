"""Offline replay for a published from-scratch ESMFold2-50M candidate."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import random

from .adapter import final_hidden_for_chains, prepare_request, request_feature_lookup
from .architecture import build_model, persistent_state_dict
from .config import EXPECTED_PARAMETER_RANGE
from .recipe import parameter_count


@contextmanager
def memory_efficient_triangle_updates(chunk_size: int):
    """Row-chunk every L x L intermediate of the reference triangle update.

    Upstream's ``set_chunk_size`` only chunks the final einsum; it still
    materializes the normalized grid, the four-way projection bundle and the
    float32 streams for the whole L x L pair grid, which is ~25 GB per update
    at 2,500 tokens (c_z = 256). This variant computes the gated streams once
    in bf16 and then produces the output in row chunks, so peak memory is a
    few pair-sized bf16 tensors instead of eight float32 ones.
    """
    import torch
    from transformers.models.esmfold2 import modeling_esmfold2_common as common

    block = common.TriangleMultiplicativeBlock
    original = block.forward

    def forward(self, pair_grid, visibility=None):
        if visibility is None:
            visibility = pair_grid.new_ones(pair_grid.shape[:-1])
        latent = self.latent_channels
        rows = pair_grid.shape[1]
        left = pair_grid.new_empty((*pair_grid.shape[:-1], latent))
        right = pair_grid.new_empty((*pair_grid.shape[:-1], latent))
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            normalized = self.norm_start(pair_grid[:, start:end])
            bundled = self.proj_bundle(normalized)
            signal, gate_logits = bundled.split(2 * latent, dim=-1)
            routed = signal * torch.sigmoid(gate_logits)
            routed = routed * visibility[:, start:end].unsqueeze(-1)
            left[:, start:end], right[:, start:end] = routed.chunk(2, dim=-1)
            del normalized, bundled, signal, gate_logits, routed
        right = right.float()
        output = torch.empty_like(pair_grid)
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            if self.flow == "outgoing":
                contracted = torch.einsum(
                    self._einsum_equation, left[:, start:end].float(), right
                )
            else:
                contracted = torch.einsum(
                    self._einsum_equation, left[:, :, start:end].float(), right
                )
            mixed = self.proj_emit(self.norm_mix(contracted).to(pair_grid.dtype))
            gate = torch.sigmoid(self.proj_gate(self.norm_start(pair_grid[:, start:end])))
            output[:, start:end] = mixed * gate
            del contracted, mixed, gate
        return output

    block.forward = forward
    try:
        yield
    finally:
        block.forward = original


class ReplayTooLarge(RuntimeError):
    """The request is larger than this runtime can fold on the available GPU."""


class ESMFold2Candidate:
    def __init__(self, model_dir: Path):
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder
        from safetensors.torch import load_file

        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "model_config.json").read_text())
        self.model = build_model()
        count = parameter_count(self.model)
        if count != int(self.config["parameter_count"]):
            raise ValueError(f"parameter audit mismatch: graph={count} package={self.config['parameter_count']}")
        if not EXPECTED_PARAMETER_RANGE[0] <= count <= EXPECTED_PARAMETER_RANGE[1]:
            raise ValueError(f"candidate parameter count {count:,} is outside 45M--55M")
        state = persistent_state_dict(
            load_file(str(self.model_dir / "model.safetensors"), device="cpu")
        )
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.fast_backend = None
        if self.device.type == "cuda":
            try:
                self.model.set_kernel_backend("fused")
                self.fast_backend = "fused"
            except Exception:
                self.model.set_kernel_backend(None)
        # Biohub's fused triangle kernels materialize whole L x L buffers and
        # exceed 80 GB from about 2,500 tokens after chain copies (2,468 replays
        # in ~13 s, 2,541 does not; the suite reaches 4,693). Above this token
        # count replay uses the reference kernels with the row-chunked triangle
        # update below, which is ~30x slower (minutes per request), so the
        # threshold sits just above the fused path's measured capacity; the
        # same path is the single retry after a CUDA out-of-memory error.
        self.low_memory_tokens = int(self.config.get("low_memory_tokens", 2480))
        self.low_memory_chunk_size = int(self.config.get("low_memory_chunk_size", 64))
        self.default_chunk_size = 64
        # Above this size even the low-memory path exceeds 80 GB (2,592 replays,
        # 2,926 does not) and each doomed attempt costs minutes, so such requests
        # are recorded as failures without a forward pass. Raise it on larger
        # GPUs (FOLDBENCH_REPLAY_MAX_TOKENS).
        self.replay_max_tokens = int(
            os.environ.get("FOLDBENCH_REPLAY_MAX_TOKENS",
                           self.config.get("replay_max_tokens", 2700))
        )
        self.builder = ESMFold2InputBuilder(
            ccd_cache=Path(self.config.get("ccd_dir", "/opt/esmfold2-assets"))
        )

    def _fold(self, features, low_memory: bool):
        import torch

        if low_memory:
            self.model.set_kernel_backend(None)
            self.model.set_chunk_size(self.low_memory_chunk_size)
            patch = memory_efficient_triangle_updates(self.low_memory_chunk_size)
        else:
            self.model.set_kernel_backend(self.fast_backend)
            self.model.set_chunk_size(self.default_chunk_size)
            patch = None
        try:
            with torch.inference_mode(), (patch or nullcontext()):
                return self.model(
                    **features,
                    num_loops=int(self.config.get("loops", 3)),
                    num_sampling_steps=int(self.config.get("sampling_steps", 20)),
                    num_diffusion_samples=1,
                    early_exit=False,
                    msa_max_depth=None,
                    msa_column_mask_rate=0.0,
                    msa_subsample_at_inference=False,
                )
        finally:
            if low_memory:
                self.model.set_kernel_backend(self.fast_backend)
                self.model.set_chunk_size(self.default_chunk_size)

    def predict(self, request, feature_paths, seed: int, sample_index: int):
        import numpy as np
        import torch

        effective_seed = int(seed) + int(sample_index)
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)

        features, chain_infos = prepare_request(
            request, self.builder, seed=effective_seed, device=self.device
        )
        for key in ("msa", "msa_attention_mask", "has_deletion", "deletion_value"):
            features.pop(key, None)
        final_hidden = final_hidden_for_chains(
            features,
            chain_infos,
            request_feature_lookup(request, feature_paths),
        ).to(self.device)
        features["lm_hidden_states"] = final_hidden.unsqueeze(2)
        tokens = int(final_hidden.shape[1])
        if tokens > self.replay_max_tokens:
            raise ReplayTooLarge(
                f"{request['request_id']}: {tokens} tokens exceeds the replay limit of "
                f"{self.replay_max_tokens} for this GPU (FOLDBENCH_REPLAY_MAX_TOKENS)"
            )
        low_memory = tokens > self.low_memory_tokens
        try:
            output = self._fold(features, low_memory=low_memory)
        except torch.OutOfMemoryError:
            if low_memory or self.device.type != "cuda":
                raise
            del features["lm_hidden_states"]
            torch.cuda.empty_cache()
            features["lm_hidden_states"] = final_hidden.unsqueeze(2)
            output = self._fold(features, low_memory=True)
        decoded = self.builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=1,
            complex_id=request["request_id"],
        )
        if isinstance(decoded, list):
            if len(decoded) != 1:
                raise RuntimeError(f"expected one structure, got {len(decoded)}")
            decoded = decoded[0]
        ptm = float(decoded.ptm or 0.0)
        iptm = None if decoded.iptm is None else float(decoded.iptm)
        ranking = ptm if iptm is None else 0.2 * ptm + 0.8 * iptm
        return {
            "mmcif": decoded.complex.to_mmcif(),
            "ranking_score": ranking,
        }


def load_model(model_dir: Path) -> ESMFold2Candidate:
    return ESMFold2Candidate(model_dir)
