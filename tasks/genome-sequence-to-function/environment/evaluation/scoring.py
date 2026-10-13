"""Prediction contract v4 and the track-prediction scorer.

The verifier's grader and the agent's validation scoring import this module, so both run the
same code on the same prediction layout (`score(..., split="valid" | "test")`).

v4 keeps every v3 metric but stores splicing predictions sparsely:

- `splice_usage_positions` int32 [K] (sorted, unique, target-window coordinates, K <= 4,096)
  with `splice_usage` [K, 734]. A hidden usage site the model did not list is a zero prediction.
- `junction_donor`, `junction_acceptor` int32 [P] (input coordinates), `junction_half` int8 [P]
  (0 = plus-strand pair, 1 = minus-strand pair; unique (half, donor, acceptor)), with
  `junctions` [P, 734], P <= 16,384. A hidden junction query whose pair was not listed is a zero
  prediction, as in v3 when an endpoint was missing.

v3 wrote the full [196608, 734] usage matrix and the [512, 512, 734] junction cube for every
example: about 600 GiB per 1,000-example test run, read twice for hashing and once for scoring,
while the metric reads only hidden positions. v4 bounds an example at about 53 MiB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

DENSE_TRACKS = {"atac": 167, "dnase": 305, "procap": 12, "cage": 546, "rna_seq": 667,
                "chip_tf": 1617, "chip_histone": 1116}
PREDICTION_SCHEMA = "genome-sequence-to-function-predictions/v4"
REPORT_SCHEMA = "genome-sequence-to-function-report/v4"
TARGET_BP = 196_608
INPUT_BP = 1_048_576
USAGE_TRACKS = JUNCTION_TRACKS = 734
MAX_USAGE_SITES = 4_096
MAX_JUNCTION_PAIRS = 16_384
FIXED_SHAPES = {**{name: (1_536, tracks) for name, tracks in DENSE_TRACKS.items()},
                "splice_sites": (TARGET_BP, 4), "contact_maps": (131_328, 28)}
MAX_PREDICTION_BYTES = 80 * 1024**3  # about 53 MiB per test example at the caps


class PerTrackMoments:
    """Streaming per-track Pearson over all scored values."""

    def __init__(self, tracks: int):
        self.n = np.zeros(tracks, np.int64)
        self.sx = np.zeros(tracks, np.float64); self.sy = self.sx.copy()
        self.sx2 = self.sx.copy(); self.sy2 = self.sx.copy(); self.sxy = self.sx.copy()
        self.missing = np.zeros(tracks, np.int64)

    def update(self, prediction, target, mask=None, missing=None):
        x = np.asarray(prediction, np.float64).reshape(-1, np.shape(prediction)[-1])
        y = np.asarray(target, np.float64).reshape(x.shape)
        valid = np.ones_like(y, bool) if mask is None else np.broadcast_to(mask, y.shape).reshape(y.shape)
        if not np.isfinite(x[valid]).all() or not np.isfinite(y[valid]).all():
            raise ValueError("non-finite metric value")
        self.n += valid.sum(0); self.sx += np.where(valid, x, 0).sum(0); self.sy += np.where(valid, y, 0).sum(0)
        self.sx2 += np.where(valid, x * x, 0).sum(0); self.sy2 += np.where(valid, y * y, 0).sum(0)
        self.sxy += np.where(valid, x * y, 0).sum(0)
        if missing is not None:
            self.missing += (np.broadcast_to(np.asarray(missing, bool)[:, None], valid.shape) & valid).sum(0)

    def update_indexed(self, prediction, target, track, missing=None):
        x = np.asarray(prediction, np.float64); y = np.asarray(target, np.float64)
        t = np.asarray(track, np.int64); size = len(self.n)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite metric value")
        self.n += np.bincount(t, minlength=size); self.sx += np.bincount(t, weights=x, minlength=size)
        self.sy += np.bincount(t, weights=y, minlength=size); self.sx2 += np.bincount(t, weights=x * x, minlength=size)
        self.sy2 += np.bincount(t, weights=y * y, minlength=size); self.sxy += np.bincount(t, weights=x * y, minlength=size)
        if missing is not None:
            self.missing += np.bincount(t, weights=np.asarray(missing, bool), minlength=size).astype(np.int64)

    def report(self):
        target_ss = np.maximum(self.n * self.sy2 - self.sy**2, 0)
        prediction_ss = np.maximum(self.n * self.sx2 - self.sx**2, 0)
        scorable = (self.n >= 2) & (target_ss > 0)
        numerator = self.n * self.sxy - self.sx * self.sy
        denominator = np.sqrt(prediction_ss * target_ss)
        pearson = np.full(len(self.n), np.nan)
        nonconstant = scorable & (denominator > 0)
        pearson[nonconstant] = numerator[nonconstant] / denominator[nonconstant]
        pearson[scorable & ~nonconstant] = 0.0
        missing_ratio = np.divide(self.missing, self.n, out=np.zeros(len(self.n)), where=self.n > 0)
        return {"mean_pearson": float(pearson[scorable].mean()) if scorable.any() else 0.0,
                "valid_tracks": int(scorable.sum()),
                "per_track_pearson": [float(v) if np.isfinite(v) else None for v in pearson],
                "values_per_track": self.n.tolist(), "missing_prediction_ratio": missing_ratio.tolist()}


def _open_target(path: Path):
    if str(path).endswith(".zarr.zip"):
        store = zarr.ZipStore(str(path), mode="r")
        return zarr.open_group(store=store, mode="r"), store
    return zarr.open_group(str(path), mode="r"), None


def _array(group, name: str, shape, dtypes, example_id: str) -> np.ndarray:
    array = group[name]
    if tuple(array.shape) != tuple(shape) or array.dtype not in dtypes:
        raise ValueError(f"{example_id}/{name} has {array.shape} {array.dtype}; expected {shape}")
    return array


_FLOATS = (np.dtype("float16"), np.dtype("float32"))
_INTS = (np.dtype("int32"), np.dtype("int64"))


def validate_example(group, example_id: str) -> None:
    """Shapes, dtypes, finiteness and sparse-key rules for one example (reads every value)."""
    for name, shape in FIXED_SHAPES.items():
        if not np.isfinite(_array(group, name, shape, _FLOATS, example_id)[:]).all():
            raise ValueError(f"{example_id}/{name} contains non-finite values")
    positions = np.asarray(group["splice_usage_positions"])
    if positions.ndim != 1 or positions.dtype not in _INTS or len(positions) > MAX_USAGE_SITES:
        raise ValueError(f"{example_id}/splice_usage_positions must be at most {MAX_USAGE_SITES} int32")
    if len(positions) and ((positions < 0).any() or (positions >= TARGET_BP).any()
                           or (np.diff(positions) <= 0).any()):
        raise ValueError(f"{example_id}/splice_usage_positions must be sorted, unique and in the target window")
    usage = _array(group, "splice_usage", (len(positions), USAGE_TRACKS), _FLOATS, example_id)
    if not np.isfinite(usage[:]).all():
        raise ValueError(f"{example_id}/splice_usage contains non-finite values")
    donor = np.asarray(group["junction_donor"]); acceptor = np.asarray(group["junction_acceptor"])
    half = np.asarray(group["junction_half"])
    pairs = len(donor)
    if (donor.ndim != 1 or acceptor.shape != donor.shape or half.shape != donor.shape
            or pairs > MAX_JUNCTION_PAIRS or donor.dtype not in _INTS or acceptor.dtype not in _INTS):
        raise ValueError(f"{example_id} junction keys must be three aligned vectors of at most {MAX_JUNCTION_PAIRS}")
    if pairs and ((donor < 0).any() or (donor >= INPUT_BP).any() or (acceptor < 0).any()
                  or (acceptor >= INPUT_BP).any() or not np.isin(half, (0, 1)).all()):
        raise ValueError(f"{example_id} junction keys are out of range")
    if len({(int(h), int(d), int(a)) for h, d, a in zip(half, donor, acceptor)}) != pairs:
        raise ValueError(f"{example_id} junction keys must be unique")
    values = _array(group, "junctions", (pairs, JUNCTION_TRACKS), _FLOATS, example_id)
    if not np.isfinite(values[:]).all():
        raise ValueError(f"{example_id}/junctions contains non-finite values")


def validate_predictions(prediction_root: Path, expected_ids: list[str]) -> None:
    group = zarr.open_group(str(prediction_root), mode="r")
    if group.attrs.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("prediction schema mismatch")
    for example_id in expected_ids:
        validate_example(group[f"examples/{example_id}"], example_id)


def _usage_prediction(pred, hidden_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    listed = np.asarray(pred["splice_usage_positions"], np.int64)
    rows = np.searchsorted(listed, hidden_positions)
    found = (rows < len(listed)) & (listed[np.minimum(rows, max(len(listed) - 1, 0))] == hidden_positions) \
        if len(listed) else np.zeros(len(hidden_positions), bool)
    values = np.zeros((len(hidden_positions), USAGE_TRACKS), np.float32)
    if found.any():
        values[found] = np.asarray(pred["splice_usage"].get_orthogonal_selection(
            (rows[found], slice(None))), np.float32)
    return values, ~found


def _junction_prediction(pred, queries: pd.DataFrame, track_halves: np.ndarray):
    keys = {(int(h), int(d), int(a)): i for i, (h, d, a) in enumerate(zip(
        np.asarray(pred["junction_half"]), np.asarray(pred["junction_donor"]),
        np.asarray(pred["junction_acceptor"])))}
    rows = np.array([keys.get((int(track_halves[int(q.track)]), int(q.donor_position),
                               int(q.acceptor_position)), -1)
                     for q in queries.itertuples(index=False)], np.int64)
    missing = rows < 0
    output = np.zeros(len(queries), np.float32)
    if (~missing).any():
        output[~missing] = np.asarray(pred["junctions"].vindex[
            rows[~missing], queries.track.to_numpy(np.int64)[~missing]], np.float32)
    return output, missing


def score(prediction_root: Path, labels: Path, split: str = "test") -> dict:
    """labels: a release root (or the grader's /labels mount) holding labels/, scoring/,
    targets/ and public/junction_tracks.parquet for `split`."""
    rows = pd.read_parquet(labels / f"labels/{split}.parquet")
    predictions = zarr.open_group(str(prediction_root), mode="r")
    track_halves = pd.read_parquet(labels / "public/junction_tracks.parquet").sort_values(
        "tensor_index").junction_channel_half.to_numpy(np.int64)
    metrics = {**{k: PerTrackMoments(v) for k, v in DENSE_TRACKS.items()},
               "splice_sites": PerTrackMoments(4), "splice_usage": PerTrackMoments(USAGE_TRACKS),
               "junctions": PerTrackMoments(JUNCTION_TRACKS), "contact_maps": PerTrackMoments(28)}
    for example_index, row in enumerate(rows.itertuples(index=False)):
        target_root, store = _open_target(labels / row.target_shard)
        target = target_root[row.target_group]
        pred = predictions[f"examples/{row.example_id}"]
        for family in DENSE_TRACKS:
            metrics[family].update(np.asarray(pred[family]), np.arcsinh(np.asarray(target[f"{family}/values"])),
                                   np.asarray(target[f"{family}/mask"], bool))
        metrics["splice_sites"].update(np.asarray(pred["splice_sites"]),
                                       np.asarray(target["splice_sites/values"], np.float32))
        hidden = np.asarray(target["splice_site_usage/positions"], np.int64)
        usage, usage_missing = _usage_prediction(pred, hidden)
        metrics["splice_usage"].update(usage, np.arcsinh(np.asarray(target["splice_site_usage/values"])),
                                       np.asarray(target["splice_site_usage/mask"], bool), usage_missing)
        queries = pd.read_parquet(labels / f"scoring/{split}" / f"part-{example_index:06d}.parquet")
        queries = queries.loc[queries.valid].reset_index(drop=True)
        junction, junction_missing = _junction_prediction(pred, queries, track_halves)
        metrics["junctions"].update_indexed(junction, np.arcsinh(queries.target_value.to_numpy(np.float32)),
                                            queries.track.to_numpy(np.int64), junction_missing)
        metrics["contact_maps"].update(np.asarray(pred["contact_maps"]),
                                       np.arcsinh(np.asarray(target["contact_maps/upper_triangle"])),
                                       np.asarray(target["contact_maps/mask"], bool))
        if store is not None:
            store.close()
    families = {k: v.report() for k, v in metrics.items()}
    overall = float(np.mean([x["mean_pearson"] for x in families.values()]))
    return {"schema": REPORT_SCHEMA, "split": split, "examples": len(rows), "families": families,
            "overall_mean_family_pearson": overall, "invalid_reason": None}
