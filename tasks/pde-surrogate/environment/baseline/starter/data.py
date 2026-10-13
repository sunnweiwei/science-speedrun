"""Training pools from the split manifest: windows of 4 context frames + 1 target, per dataset."""
import json
import random
from pathlib import Path

import numpy as np

from pde_io import family_of, read_window, trajectory_length


class Pools:
    def __init__(self, data_dir, manifest, split="train", context=4, datasets=None, max_windows_per_traj=None,
                 allow_missing=False):
        """Index every pool of `split`. A pool the manifest names but the disk lacks raises unless
        allow_missing=True, in which case it is recorded in self.missing: the first baseline was
        trained with one dataset and 50 shards silently absent, and nothing recorded it."""
        man = json.loads(Path(manifest).read_text())
        self.context, self.by_dataset, self.missing = context, {}, []
        for pool in man["pools"][split]:
            if datasets and pool["dataset"] not in datasets:
                continue
            fam = family_of(pool["source"])
            path = Path(data_dir) / pool["source"] / pool["file"]
            if not path.exists():
                self.missing.append(f"{pool['source']}/{pool['file']}")
                continue
            length = trajectory_length(fam, path)
            ts = list(range(0, length - context))
            if max_windows_per_traj:
                ts = ts[:max_windows_per_traj]
            entries = [(fam, str(path), traj, t) for traj in range(*pool["trajectories"]) for t in ts]
            self.by_dataset.setdefault(pool["dataset"], []).extend(entries)
        if self.missing and not allow_missing:
            raise FileNotFoundError(f"{len(self.missing)} pool file(s) of split '{split}' are not under {data_dir}: "
                                    f"{self.missing[:3]}{' ...' if len(self.missing) > 3 else ''}. Materialize them, "
                                    f"or pass allow_missing=True (train.py --allow-missing-pools) to train without them.")
        self.datasets = sorted(self.by_dataset)

    def sample(self, dataset, n, rng):
        pool = self.by_dataset[dataset]
        xs, ys = [], []
        for fam, path, traj, t in rng.sample(pool, min(n, len(pool))):
            frames, _ = read_window(fam, path, traj, t, self.context, 1)
            xs.append(frames[: self.context]); ys.append(frames[self.context])
        return np.stack(xs), np.stack(ys)   # (n, T, C, X, Y), (n, C, X, Y)

    def normalization(self, n_frames=64, seed=0):
        """Per-dataset, per-channel mean and std over sampled training frames."""
        rng = random.Random(seed)
        stats = {}
        for ds in self.datasets:
            x, _ = self.sample(ds, max(1, n_frames // self.context), rng)
            flat = x.transpose(2, 0, 1, 3, 4).reshape(x.shape[2], -1)
            stats[ds] = {"mean": flat.mean(1).tolist(), "std": (flat.std(1) + 1e-6).tolist()}
        return stats
