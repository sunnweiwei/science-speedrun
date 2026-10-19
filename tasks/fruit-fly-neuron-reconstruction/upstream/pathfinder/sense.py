"""Independent SENSE prototype; architectural choices are logged in REPRODUCTION.md."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.first = nn.Conv3d(incoming, outgoing, 3, padding=1)
        self.second = nn.Conv3d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv3d(incoming, outgoing, 1)

    def forward(self, x):
        return self.skip(x) + self.second(F.silu(self.first(F.silu(x))))


class Sense(nn.Module):
    def __init__(self, width=8, factors=(1, 2, 4, 8, 8)):
        super().__init__()
        if width < 1 or not factors or min(factors) < 1:
            raise ValueError('Positive width and nonempty positive factors required')
        self.levels = len(factors)
        self.embed = nn.Conv3d(2, width, 1)
        self.down = nn.ModuleList(); self.after_pool = nn.ModuleList()
        incoming = width
        widths = [width * f for f in factors]
        for outgoing in widths:
            self.down.append(nn.Sequential(Residual(incoming, outgoing), Residual(outgoing, outgoing), Residual(outgoing, outgoing)))
            self.after_pool.append(Residual(outgoing, outgoing))
            incoming = outgoing
        self.middle = nn.Sequential(Residual(incoming, incoming), Residual(incoming, incoming))
        self.up_before = nn.ModuleList(); self.up_after = nn.ModuleList()
        for outgoing in reversed(widths):
            self.up_before.append(Residual(incoming + outgoing, outgoing))
            # Independent interpretation: expand channels 2x on the decoder, then
            # project back in the next skip block. No author weight compatibility.
            self.up_after.append(Residual(outgoing, 2 * outgoing))
            incoming = 2 * outgoing
        self.out = nn.Conv3d(incoming, 1, 3, padding=1)

    def forward(self, raw, prompt):
        if raw.shape != prompt.shape or raw.ndim != 5 or raw.shape[1] != 1:
            raise ValueError('raw and prompt must be matching B,1,Z,Y,X tensors')
        if any(n % (2**self.levels) for n in raw.shape[-3:]):
            raise ValueError('Spatial dimensions must be divisible by the downsampling factor')
        x = self.embed(torch.cat((raw, prompt), dim=1)); skips = []
        for block, after in zip(self.down, self.after_pool):
            x = block(x)
            x = F.avg_pool3d(F.silu(x), 2)
            skips.append(x)
            x = after(F.silu(x))
        x = self.middle(x)
        for before, after, skip in zip(self.up_before, self.up_after, reversed(skips)):
            x = before(torch.cat((x, skip), dim=1))
            x = after(F.interpolate(x, scale_factor=2, mode='nearest'))
        return self.out(F.silu(x))


def prompt_mask(labels, seed_id, radius=16):
    """Probability prompt restricted to a central voxel-radius sphere."""
    if labels.ndim != 3 or labels.dtype.kind not in 'ui' or seed_id <= 0 or radius <= 0:
        raise ValueError('Require positive seed/radius and 3D integer labels')
    center = np.asarray(labels.shape) // 2
    if int(labels[tuple(center)]) != seed_id:
        raise ValueError('Crop center must lie in the seed segment')
    grid = np.ogrid[tuple(slice(0, n) for n in labels.shape)]
    inside = sum((a - c)**2 for a, c in zip(grid, center)) <= radius**2
    return np.where(inside, np.where(labels == seed_id, .95, .05), .5).astype(np.float32)


def candidates(probability, segments, full_sizes, seed_id, min_size=500, threshold=.5):
    """Overlap denominator uses full segment size, not visible crop size."""
    if probability.shape != segments.shape or segments.dtype.kind not in 'ui':
        raise ValueError('Aligned probability and integer segments required')
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError('Probabilities must be finite in [0,1]')
    if not 0 <= threshold <= 1 or min_size < 1:
        raise ValueError('Invalid candidate threshold/size')
    ids, visible = np.unique(segments, return_counts=True)
    bulk_sizes=getattr(full_sizes,'sizes_for',None)
    if callable(bulk_sizes):
        positive=ids!=0
        try:sizes=bulk_sizes(ids[positive])
        except KeyError as error:raise ValueError('Full segment sizes missing or smaller than visible crop') from error
        if np.any(sizes<visible[positive]):raise ValueError('Full segment sizes missing or smaller than visible crop')
        # Exact integer-to-float conversion within the supported volume range.
        # Retain scalar arithmetic for unusually large external size tables.
        if np.all(sizes<=2**53):
            selected,counts=np.unique(segments[probability>.5],return_counts=True)
            keep=(selected!=0)&(selected!=seed_id);selected=selected[keep];counts=counts[keep]
            sizes=bulk_sizes(selected);scores=counts/sizes
            eligible=(sizes>=min_size)&(scores>=threshold)
            result=[dict(source=int(seed_id),target=int(sid),overlap=float(score)) for sid,score in zip(selected[eligible],scores[eligible])]
            return sorted(result,key=lambda r:(-r['overlap'],r['target']))
    for sid, count in zip(ids, visible):
        if sid and (int(sid) not in full_sizes or full_sizes[int(sid)] < count):
            raise ValueError('Full segment sizes missing or smaller than visible crop')
    selected, counts = np.unique(segments[probability > .5], return_counts=True)
    result = []
    for sid, n in zip(selected, counts):
        sid = int(sid)
        if sid in (0, seed_id): continue
        size = full_sizes[sid]; score = int(n) / size
        if size >= min_size and score >= threshold:
            result.append(dict(source=int(seed_id), target=sid, overlap=score))
    return sorted(result, key=lambda r: (-r['overlap'], r['target']))
