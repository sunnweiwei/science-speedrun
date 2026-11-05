#!/usr/bin/env python3
"""Compare both towers to pinned author Uni-Mol/Uni-Core source on CPU.

Source root must contain `author_encoder.py` from Drug-The-Whole-Genome
9d0db739 and `unicore/modules/{transformer_encoder_layer,multihead_attention,
layer_norm,softmax_dropout}.py` from Uni-Core revision
ace6fae1c8479a9751f2bb1e1d6e4047427bc134. This optional
integration test keeps upstream source out of the submitted model package.
"""

import argparse
from pathlib import Path
import sys

import torch

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))

from unimol_encoder import UniMolEncoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--molecule-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from author_encoder import TransformerEncoderWithPair

    for kind, vocab in (("molecule", 31), ("pocket", 10)):
        torch.manual_seed(17)
        ours = UniMolEncoder(vocab).eval()
        if kind == "molecule":
            ours.load_official_checkpoint(
                args.molecule_checkpoint,
                "da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9")
        official = TransformerEncoderWithPair(
            encoder_layers=15, embed_dim=512, ffn_embed_dim=2048,
            attention_heads=64, no_final_head_layer_norm=True).eval()
        official.load_state_dict(ours.encoder.state_dict(), strict=True)
        tokens = torch.tensor([[1, 4, 5, 2, 0], [1, 4, 6, 7, 2]])
        distances = torch.rand(2, 5, 5)
        edges = tokens[:, :, None] * vocab + tokens[:, None, :]
        batch = {"tokens": tokens, "distances": distances, "edge_types": edges}
        with torch.no_grad():
            actual = ours(batch)
            bias = ours.gbf_proj(ours.gbf(distances, edges))
            bias = bias.permute(0, 3, 1, 2).contiguous().view(-1, 5, 5)
            expected = official(ours.embed_tokens(tokens), attn_mask=bias,
                                padding_mask=tokens.eq(0))[0][:, 0]
        error = (actual - expected).abs().max().item()
        print(f"{kind}: max absolute error {error}")
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    main()
