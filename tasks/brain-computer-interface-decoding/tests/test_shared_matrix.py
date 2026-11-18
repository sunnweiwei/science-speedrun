"""One backbone per authorized family must receive gradients from each of the four tracks."""
import argparse
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

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'environment/baseline/braindecode'))
try:
    from model import MODELS, SPECS
    from pretrained import load_backbone
    from shared import SharedModel
except ModuleNotFoundError as exc:
    if __name__ == '__main__':
        raise
    raise unittest.SkipTest(f'Requires braindecode container runtime: {exc}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3407)
    for family in MODELS:
        model = SharedModel(family).to(device)
        loaded = load_backbone(model, family, args.checkpoint_dir)
        for track, (channels, seconds, classes) in SPECS.items():
            x = torch.randn(1, channels, seconds, 200, device=device)
            logits = model(track, x)
            assert logits.shape == (1, classes)
            logits.sum().backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.backbone.parameters())
            model.zero_grad(set_to_none=True)
            print(json.dumps({'family': family, 'track': track, 'status': 'ok',
                              'initialization': loaded['initialization']}), flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
