"""Channel-shared convolutional surrogate: 4 context frames of ONE channel -> next frame of that channel.

Every physical channel is processed independently with shared weights, so the
same network handles datasets with 1 to 11 channels on any grid size. It sees no
metadata and no cross-channel coupling; both are deliberate omissions for an
agent to improve on.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class UNet(nn.Module):
    def __init__(self, context=4, width=32, levels=3):
        super().__init__()
        self.context = context
        ws = [width * 2 ** i for i in range(levels)]
        self.inc = block(context, ws[0])
        self.down = nn.ModuleList([block(ws[i], ws[i + 1]) for i in range(levels - 1)])
        self.up = nn.ModuleList([block(ws[i + 1] + ws[i], ws[i]) for i in reversed(range(levels - 1))])
        self.out = nn.Conv2d(ws[0], 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """x: (N, context, X, Y) normalized frames of one channel -> (N, 1, X, Y) next frame."""
        h, skips = self.inc(x), []
        for d in self.down:
            skips.append(h); h = d(F.avg_pool2d(h, 2))
        for u in self.up:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = u(torch.cat([h, s], dim=1))
        return x[:, -1:] + self.out(h)          # residual on the last context frame

    @torch.no_grad()
    def rollout(self, context, steps):
        """context: (C, T, X, Y) normalized -> (steps, C, X, Y) normalized predictions."""
        frames = list(context.unbind(1))
        out = []
        for _ in range(steps):
            x = torch.stack(frames[-self.context:], dim=1)
            nxt = self.forward(x)[:, 0]
            out.append(nxt); frames.append(nxt)
        return torch.stack(out, dim=0)
