"""Source-balanced window sampling for the packed corpus.

The packed corpus is dominated by a few huge directories (ERA5, CMIP6 and buildings_900k hold
about 85 % of the series). Sampling uniformly over series would make the model mostly a climate
and building-load forecaster. Instead a window is drawn in three stages:

1. a source group, with fixed weights (`GROUP_WEIGHTS`);
2. a directory within the group, with weight proportional to sqrt(points in the directory)
   (temperature sampling, so small distinctive datasets are not drowned out);
3. a series within the directory, uniformly; then a random window as in `WindowSampler`.

Synthetic KernelSynth series are mixed in with `synthetic_fraction`, as in the starter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUP_WEIGHTS = {"diverse": 0.50, "climate": 0.25, "buildings": 0.15, "cloud": 0.10}


def group_of(directory: str) -> str:
    if directory.startswith(("era5_", "cmip6_")):
        return "climate"
    if directory == "buildings_900k":
        return "buildings"
    if directory == "borg_cluster_data_2011":
        return "cloud"
    return "diverse"


class MixtureSampler:
    def __init__(self, packed: Path, context: int, horizon: int, seed: int = 0, min_context: int = 64,
                 synthetic=None, synthetic_fraction: float = 0.0, group_weights: dict | None = None, temperature: float = 0.5):
        self.values = np.load(packed / "values.npy", mmap_mode="r")
        self.offsets = np.load(packed / "offsets.npy")
        self.lengths = np.diff(self.offsets)
        self.context, self.horizon, self.min_context = context, horizon, min_context
        self.rng = np.random.default_rng(seed)
        self.synthetic, self.synthetic_fraction = synthetic, synthetic_fraction
        meta = json.loads((packed / "meta.json").read_text())
        names = np.array([s["dataset"] for s in meta["series"]])
        ok = self.lengths >= min_context + horizon
        self.candidates = np.nonzero(ok)[0]
        by_dir = {}
        for d in np.unique(names[ok]):
            idx = np.nonzero(ok & (names == d))[0]
            by_dir[str(d)] = idx
        weights = dict(group_weights or GROUP_WEIGHTS)
        groups = {g: [d for d in by_dir if group_of(d) == g] for g in weights}
        groups = {g: ds for g, ds in groups.items() if ds}
        total = sum(weights[g] for g in groups)
        self.dirs, probs = [], []
        for g, ds in groups.items():
            pts = np.array([self.lengths[by_dir[d]].sum() for d in ds], dtype=np.float64)
            w = pts ** temperature
            w = w / w.sum() * weights[g] / total
            self.dirs += ds
            probs += list(w)
        self.dir_index = [by_dir[d] for d in self.dirs]
        self.dir_probs = np.array(probs) / np.sum(probs)
        self.group_share = {g: float(sum(p for d, p in zip(self.dirs, self.dir_probs) if group_of(d) == g)) for g in groups}

    def sample(self, batch: int):
        ctx = np.full((batch, self.context), np.nan, np.float32)
        tgt = np.full((batch, self.horizon), np.nan, np.float32)
        dirs = self.rng.choice(len(self.dirs), size=batch, p=self.dir_probs)
        for b in range(batch):
            if self.synthetic is not None and self.rng.random() < self.synthetic_fraction:
                s = self.synthetic(self.rng, self.context + self.horizon)
                ctx[b], tgt[b] = s[:self.context], s[self.context:]
                continue
            pool = self.dir_index[dirs[b]]
            i = pool[self.rng.integers(len(pool))]
            start, length = self.offsets[i], self.lengths[i]
            n_ctx = int(self.rng.integers(self.min_context, min(self.context, length - self.horizon) + 1))
            cut = int(self.rng.integers(n_ctx, length - self.horizon + 1))
            win = self.values[start + cut - n_ctx: start + cut + self.horizon]
            ctx[b, self.context - n_ctx:] = win[:n_ctx]
            tgt[b] = win[n_ctx:]
        return ctx, tgt
