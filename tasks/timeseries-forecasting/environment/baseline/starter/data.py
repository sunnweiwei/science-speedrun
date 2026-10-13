#!/usr/bin/env python3
"""Corpus packing and window sampling for the starter.

    python3 data.py pack --data /workspace/data/gifteval_pretrain --out /workspace/research/packed [--tier fast|full] [--dirs a,b]

reads every allow-listed GiftEvalPretrain directory (Arrow `save_to_disk`
shards; multivariate targets are flattened to one series per variate), drops
series shorter than MIN_LENGTH, and writes a flat float32 memmap
(`values.npy`), int64 `offsets.npy` and `meta.json` (per-series dataset,
frequency, length). Training samples random (context, horizon) windows from
the memmap, so the whole corpus never has to fit in RAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MIN_LENGTH = 64


def iter_arrow_series(directory: Path):
    import pyarrow as pa

    for shard in sorted(directory.glob("data-*.arrow")):
        with open(shard, "rb") as f:
            table = pa.ipc.open_stream(f).read_all()
        freq = table.column("freq")[0].as_py() if table.num_rows else ""
        target = table.column("target").combine_chunks()
        vt = target.type.value_type
        if pa.types.is_list(vt) or pa.types.is_fixed_size_list(vt):
            inner = target.values
            inner_off = inner.offsets.to_numpy()
            flat = inner.values.to_numpy(zero_copy_only=False).astype(np.float32)
            for j in range(len(inner)):
                yield freq, flat[inner_off[j]:inner_off[j + 1]]
        else:
            off = target.offsets.to_numpy()
            flat = target.values.to_numpy(zero_copy_only=False).astype(np.float32)
            for i in range(len(target)):
                yield freq, flat[off[i]:off[i + 1]]


def pack(data_root: Path, out: Path, tier: str, dirs: list[str] | None, lock_path: Path | None):
    if dirs is None:
        if lock_path and lock_path.exists():
            lock = json.loads(lock_path.read_text())
            d = next(s for s in lock["sources"] if s["id"] == "gifteval_pretrain")["directories"]
            dirs = sorted(k for k, v in d.items() if tier == "full" or v["tier"] == "fast")
        else:
            dirs = sorted(p.name for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    out.mkdir(parents=True, exist_ok=True)
    # first pass: total length
    t0 = time.time()
    total, meta = 0, []
    for d in dirs:
        if not (data_root / d).exists():
            print(f"  skip {d}: not materialized", file=sys.stderr); continue
        for freq, s in iter_arrow_series(data_root / d):
            if len(s) < MIN_LENGTH or not np.isfinite(s).any():
                continue
            meta.append((d, freq, len(s))); total += len(s)
        print(f"  {d:40s} cumulative {len(meta):9d} series {total/1e9:.3f} G points ({time.time()-t0:.0f}s)", file=sys.stderr)
    values = np.lib.format.open_memmap(out / "values.npy", mode="w+", dtype=np.float32, shape=(total,))
    offsets = np.zeros(len(meta) + 1, dtype=np.int64)
    pos, k = 0, 0
    for d in dirs:
        if not (data_root / d).exists():
            continue
        for freq, s in iter_arrow_series(data_root / d):
            if len(s) < MIN_LENGTH or not np.isfinite(s).any():
                continue
            values[pos:pos + len(s)] = s; pos += len(s); k += 1; offsets[k] = pos
    values.flush()
    np.save(out / "offsets.npy", offsets)
    (out / "meta.json").write_text(json.dumps({"n_series": len(meta), "n_points": int(total), "dirs": dirs,
                                               "series": [{"dataset": m[0], "freq": m[1], "length": m[2]} for m in meta]}))
    print(f"packed {len(meta)} series, {total/1e9:.3f} G points into {out} in {time.time()-t0:.0f}s", file=sys.stderr)


class WindowSampler:
    """Uniform-over-series sampling of (context, target) windows from a packed corpus (plus optional synthetic series)."""

    def __init__(self, packed: Path, context: int, horizon: int, seed: int = 0, min_context: int = 64, synthetic=None, synthetic_fraction: float = 0.0):
        self.values = np.load(packed / "values.npy", mmap_mode="r")
        self.offsets = np.load(packed / "offsets.npy")
        self.lengths = np.diff(self.offsets)
        self.context, self.horizon, self.min_context = context, horizon, min_context
        self.rng = np.random.default_rng(seed)
        self.synthetic, self.synthetic_fraction = synthetic, synthetic_fraction
        ok = self.lengths >= min_context + horizon
        self.candidates = np.nonzero(ok)[0]
        if len(self.candidates) == 0:
            raise RuntimeError("no series long enough for the requested windows")

    def sample(self, batch: int):
        """Returns ctx (B, context) float32 with NaN padding on the left, tgt (B, horizon) float32 (NaN = missing)."""
        ctx = np.full((batch, self.context), np.nan, np.float32)
        tgt = np.full((batch, self.horizon), np.nan, np.float32)
        for b in range(batch):
            if self.synthetic is not None and self.rng.random() < self.synthetic_fraction:
                s = self.synthetic(self.rng, self.context + self.horizon)
                ctx[b], tgt[b] = s[:self.context], s[self.context:]
                continue
            i = self.candidates[self.rng.integers(len(self.candidates))]
            start, length = self.offsets[i], self.lengths[i]
            n_ctx = int(self.rng.integers(self.min_context, min(self.context, length - self.horizon) + 1))
            cut = int(self.rng.integers(n_ctx, length - self.horizon + 1))
            win = self.values[start + cut - n_ctx: start + cut + self.horizon]
            ctx[b, self.context - n_ctx:] = win[:n_ctx]
            tgt[b] = win[n_ctx:]
        return ctx, tgt


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--tier", default="full", choices=["full", "fast"]); p.add_argument("--dirs", default="")
    p.add_argument("--lock", default=str(Path(__file__).resolve().parents[2] / "data" / "sources.lock.json"))
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(Path(args.data), Path(args.out), args.tier, args.dirs.split(",") if args.dirs else None, Path(args.lock))


if __name__ == "__main__":
    main()
