"""All authorized Braindecode model/track combinations must train one update from scratch."""
import json
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if __name__ == '__main__':
        raise
    raise unittest.SkipTest(f'Requires PyTorch container runtime: {exc}')

local_baseline = Path(__file__).resolve().parents[1] / 'environment' / 'baseline' / 'braindecode'
container_baseline = Path('/workspace/baseline/braindecode')
sys.path.insert(0, str(local_baseline if local_baseline.exists() else container_baseline))
try:
    from model import MODELS, SPECS, Model
except ModuleNotFoundError as exc:
    if __name__ == '__main__':
        raise
    raise unittest.SkipTest(f'Requires braindecode container runtime: {exc}')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3407)
    for family in MODELS:
        for track, (channels, seconds, classes) in SPECS.items():
            model = Model(track, family).to(device)
            model.train()
            x = torch.randn(2, channels, seconds, 200, device=device)
            y = torch.tensor([0, classes - 1], device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            assert tuple(logits.shape) == (2, classes), (family, track, logits.shape)
            loss = torch.nn.functional.cross_entropy(logits, y)
            assert torch.isfinite(loss), (family, track)
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), (family, track)
            optimizer.step()
            print(json.dumps({'model': family, 'track': track, 'status': 'ok',
                              'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
            del model, optimizer, x, y, logits, loss
            if device == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
