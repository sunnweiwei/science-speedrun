"""Bounded predicted FFN instances -> full-field SENSE candidate edges.

No ground truth is accepted. Crops use real EM context beyond the segmentation
ROI; unknown predicted labels outside the ROI receive neutral prompts. One
eligible center per instance is the baseline, not paper ensemble coverage.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from sense import Sense, prompt_mask, candidates
from infer_sense import checkpoint_config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, device_guard, sha256, write_json


def validate_geometry(raw, segmentation, receipt):
    if receipt.get('ground_truth_used') is not False:
        raise ValueError('Require an EM-only FFN provenance receipt')
    if raw.ndim != 3 or raw.dtype != np.uint8 or segmentation.ndim != 3 or segmentation.dtype.kind != 'u':
        raise ValueError('Require 3D uint8 raw and unsigned instance labels')
    roi = np.asarray(receipt.get('roi'))
    if roi.shape != (6,) or roi.dtype.kind not in 'iu':
        raise ValueError('Integer FFN ROI required')
    start, stop = roi[::2], roi[1::2]
    if (start < 0).any() or (stop > raw.shape).any() or not np.array_equal(stop - start, segmentation.shape):
        raise ValueError('FFN ROI and segmentation/raw shapes disagree')
    meta = receipt.get('metadata', {})
    spacing = np.asarray(meta.get('voxel_size_nm'), dtype=float)
    origin = np.asarray(meta.get('origin_nm'), dtype=float)
    raw_origin = np.asarray(meta.get('raw_origin_nm'), dtype=float)
    if (meta.get('axes') != 'zyx' or meta.get('coordinate_convention') != 'voxel_outer_corner'
        or spacing.shape != (3,) or origin.shape != (3,) or raw_origin.shape != (3,)
        or not np.isfinite([spacing, origin, raw_origin]).all() or (spacing <= 0).any()
        or not np.allclose(origin, raw_origin + start * spacing)
        or list(meta.get('shape', [])) != list(segmentation.shape)):
        raise ValueError('Explicit consistent FFN physical metadata required')
    return start, meta


def candidate_pass(raw, segmentation, receipt, predict, *, seconds, max_calls, min_size=500, threshold=.5):
    """predict(raw_float32, probability_prompt) returns one 128^3 probability array."""
    if seconds <= 0 or max_calls < 0 or min_size < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid candidate budgets/threshold')
    started = time.monotonic()
    start, meta = validate_geometry(raw, segmentation, receipt)
    ids, counts = np.unique(segmentation, return_counts=True)
    sizes = {int(i): int(n) for i, n in zip(ids, counts) if i}
    covered, boundary, edges = [], [], {}
    calls = 0
    exhausted = False
    # Spatial indexing is CPU-only, deterministic, and never refers to GT.
    # Only a single instance's coordinates are retained at a time.
    for sid in sorted(sizes):
        if calls >= max_calls or time.monotonic() - started >= seconds:
            exhausted = True
            break
        locations = np.argwhere(segmentation == sid)
        global_locations = locations + start
        eligible = ((global_locations >= 64) & (global_locations <= np.asarray(raw.shape) - 64)).all(axis=1)
        locations = locations[eligible]
        if not len(locations):
            boundary.append(sid)
            continue
        midpoint = locations.mean(axis=0)
        center = locations[np.argmin(((locations - midpoint)**2).sum(axis=1))]
        global_center = center + start
        raw_slice = tuple(slice(int(v - 64), int(v + 64)) for v in global_center)
        crop = np.array(raw[raw_slice], dtype=np.float32) / 255
        labels = np.zeros((128, 128, 128), dtype=segmentation.dtype)
        crop_start = center - 64
        lo = np.maximum(crop_start, 0); hi = np.minimum(crop_start + 128, segmentation.shape)
        local = tuple(slice(int(l - c), int(h - c)) for l, h, c in zip(lo, hi, crop_start))
        source = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        labels[local] = segmentation[source]
        prompt = prompt_mask(labels, sid, radius=16)
        known = np.zeros(labels.shape, bool); known[local] = True
        prompt[(~known) | (labels == 0)] = .5
        if time.monotonic() - started >= seconds:
            exhausted = True
            break
        probability = predict(crop, prompt)
        calls += 1; covered.append(sid)
        for row in candidates(probability, labels, sizes, sid, min_size=min_size, threshold=threshold):
            key = (row['source'], row['target'])
            if key not in edges or row['overlap'] > edges[key]['overlap']:
                edges[key] = dict(row, sense=row['overlap'], center_zyx=center.tolist())
    metadata = {k: meta[k] for k in ('axes', 'voxel_size_nm', 'origin_nm', 'shape', 'coordinate_convention')}
    result = dict(schema='pathfinder-candidate-edges/v1', edges=list(edges.values()), metadata=metadata,
                  ground_truth_used=False, paper_comparable=False)
    elapsed = time.monotonic() - started
    audit = dict(calls=calls, seconds=elapsed, max_calls=max_calls, seconds_limit=seconds,
                 time_limit_overrun_seconds=max(0.0, elapsed-seconds),
                 time_limit_policy='checked between CPU preprocessing and whole model calls; in-flight call is not interrupted',
                 budget_exhausted=exhausted, predicted_instances=len(sizes), covered_instance_ids=covered,
                 skipped_boundary_instance_ids=boundary, unprocessed_instance_ids=sorted(set(sizes)-set(covered)-set(boundary)),
                 instance_coverage=len(covered)/len(sizes) if sizes else 0.0,
                 empty_segmentation=not bool(sizes), full_sizes=sizes, field_of_view=[128]*3,
                 centers_per_instance=1, denominator='entire supplied predicted instance in FFN ROI',
                 ground_truth_used=False, paper_comparable=False, upstream_ffn_budget_exhausted=receipt.get('budget_exhausted'))
    return result, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--ffn-output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=1800)
    p.add_argument('--max-calls', type=int, default=256)
    p.add_argument('--min-size', type=int, default=500)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    a = p.parse_args(); device_guard(a.device); torch.set_num_threads(2)
    receipt_path = a.ffn_output / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if sha256(a.raw) != receipt.get('raw_sha256'):
        raise ValueError('Raw image hash does not match FFN input')
    raw = np.load(a.raw, mmap_mode='r', allow_pickle=False)
    seg_path = a.ffn_output / 'segmentation.npy'
    segmentation = np.load(seg_path, mmap_mode='r', allow_pickle=False)
    _, meta = validate_geometry(raw, segmentation, receipt)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint_config(checkpoint, meta['voxel_size_nm'])
    if list(config.get('input_shape', [])) != [128]*3 or config.get('prompt_radius') != 16:
        raise ValueError('This driver requires the full128 SENSE recipe and radius16 prompt')
    model = Sense(config['width'], tuple(config['factors'])).to(a.device)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    def predict(crop, prompt):
        def tensor(array): return torch.from_numpy(array)[None,None].to(a.device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.device=='cuda'):
            return model(tensor(crop), tensor(prompt)).sigmoid()[0,0].float().cpu().numpy()
    result, audit = candidate_pass(raw, segmentation, receipt, predict, seconds=a.seconds,
        max_calls=a.max_calls, min_size=a.min_size, threshold=a.threshold)
    audit.update(raw_sha256=receipt['raw_sha256'], segmentation_sha256=sha256(seg_path),
        ffn_receipt_sha256=sha256(receipt_path), checkpoint_sha256=sha256(a.checkpoint))
    out = external(a.output); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'edges.json', result); write_json(out/'receipt.json', audit)


if __name__ == '__main__':
    main()
