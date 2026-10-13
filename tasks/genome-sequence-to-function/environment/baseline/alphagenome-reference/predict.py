#!/usr/bin/env python3
"""Stream the single benchmark model into the sharded prediction contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from numcodecs import Blosc
import numpy as np
import torch
import zarr

from prediction_contract import DENSE_TRACKS, MAX_JUNCTION_PAIRS, MAX_USAGE_SITES, PREDICTION_SCHEMA
from reference.alphagenome_50m.dataset import AlphaGenomeDataset
from reference.alphagenome_50m.model import CompactModel, ModelConfig


COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)


def main(input_root: Path, output_root: Path, checkpoint: Path, split: str = "test") -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["config"])
    model = CompactModel(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = AlphaGenomeDataset(input_root, split, include_targets=False)
    model.set_junction_track_halves(dataset.junction_track_halves)
    output_root.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_root / "predictions.zarr"), mode="w")
    ids = []
    with torch.inference_mode():
        for sample in dataset:
            example_id = sample["example_id"]
            dna = sample["dna"][None].to(device)
            trunk = model._trunk(dna)
            dense_features = model._dense_features(trunk)
            group = root.create_group(f"examples/{example_id}")
            for family in DENSE_TRACKS:
                values = model.dense_heads[family](dense_features)[0]
                group.create_dataset(family, data=values.cpu().numpy().astype(np.float16), compressor=COMPRESSOR)

            all_site_logits = model._splice_site_logits(trunk)
            target_start = (config.input_length - config.target_length) // 2
            group.create_dataset(
                "splice_sites",
                data=all_site_logits[0, target_start:target_start + config.target_length].cpu().numpy().astype(np.float16),
                chunks=(4096, 4), compressor=COMPRESSOR,
            )
            predicted_positions = model.select_junction_positions(all_site_logits)[0]

            # Usage at the model's own predicted sites (v4 is sparse; unlisted hidden sites score 0).
            usage_bins = model.splice_usage_head(dense_features)[0].float()
            sites = predicted_positions.cpu().numpy().astype(np.int64).reshape(-1)
            sites = np.unique(sites[(sites >= target_start) & (sites < target_start + config.target_length)]
                              - target_start)[:MAX_USAGE_SITES]
            group.create_dataset("splice_usage_positions", data=sites.astype(np.int32), compressor=COMPRESSOR)
            usage = usage_bins[torch.as_tensor(sites // 128, device=usage_bins.device)] if len(sites) \
                else usage_bins[:0]
            group.create_dataset("splice_usage", data=usage.cpu().numpy().astype(np.float16),
                                 chunks=(1024, 734), compressor=COMPRESSOR)

            # Junctions: score every donor x acceptor pair of each strand, keep the strongest pairs.
            positions_batch = predicted_positions[None]
            site_factors = model.junction_site_factors(trunk, positions_batch)
            halves = model.junction_track_half
            candidates = []
            for row_start in range(0, 512, 128):
                for column_start in range(0, 512, 128):
                    block = torch.cat([
                        model.junction_block(
                            trunk, positions_batch, row_start, row_start + 128,
                            column_start, column_start + 128, track_start,
                            min(track_start + 64, config.junction_tracks), site_factors,
                        )[0]
                        for track_start in range(0, config.junction_tracks, 64)
                    ], dim=-1).float()  # [128, 128, 734]
                    for half in (0, 1):
                        donors = predicted_positions[2 * half, row_start:row_start + 128]
                        acceptors = predicted_positions[2 * half + 1, column_start:column_start + 128]
                        valid = (donors[:, None] >= 0) & (acceptors[None, :] >= 0)
                        strength = block[..., halves == half].amax(dim=-1).masked_fill(~valid, -torch.inf)
                        keep = min(MAX_JUNCTION_PAIRS // 2, int(valid.sum()))
                        if not keep:
                            continue
                        best = strength.flatten().topk(keep)
                        rows, columns = best.indices // 128, best.indices % 128
                        candidates.append((best.values, torch.full_like(rows, half), donors[rows],
                                           acceptors[columns], block[rows, columns]))
            if candidates:
                score, half, donor, acceptor, values = (torch.cat(x) for x in zip(*candidates))
                order = score.topk(min(MAX_JUNCTION_PAIRS, len(score))).indices
                half, donor, acceptor, values = half[order], donor[order], acceptor[order], values[order]
            else:
                half = donor = acceptor = torch.zeros(0, dtype=torch.long)
                values = torch.zeros(0, config.junction_tracks)
            group.create_dataset("junction_half", data=half.cpu().numpy().astype(np.int8), compressor=COMPRESSOR)
            group.create_dataset("junction_donor", data=donor.cpu().numpy().astype(np.int32), compressor=COMPRESSOR)
            group.create_dataset("junction_acceptor", data=acceptor.cpu().numpy().astype(np.int32), compressor=COMPRESSOR)
            group.create_dataset("junctions", data=values.cpu().numpy().astype(np.float16),
                                 chunks=(4096, 734), compressor=COMPRESSOR)

            contact = trunk.reshape(
                1, config.contact_bins, -1, config.hidden_dim
            ).mean(dim=2)
            q, k = model.contact_q(contact), model.contact_k(contact)
            row, column = model.contact_row, model.contact_column
            contact_values = 0.5 * (q[:, row] * k[:, column] + q[:, column] * k[:, row])
            group.create_dataset(
                "contact_maps", data=contact_values[0].cpu().numpy().astype(np.float16),
                chunks=(8192, 28), compressor=COMPRESSOR,
            )
            ids.append(example_id)
            print(json.dumps({"event": "predict", "example": len(ids), "total": len(dataset)}), flush=True)
    root.attrs.update({"schema": PREDICTION_SCHEMA, "example_ids": ids})
    (output_root / "example_ids.json").write_text(json.dumps(ids) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "last.pt")
    parser.add_argument("--split", default="test", choices=("valid", "test"),
                        help="manifest to predict; the verifier uses test, local validation uses valid")
    args = parser.parse_args()
    main(args.input_root, args.output_root, args.checkpoint, args.split)
