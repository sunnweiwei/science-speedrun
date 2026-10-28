"""Portable Braindecode model inference; the package is self-contained."""
import argparse
import json
from pathlib import Path
import torch
from data import Dataset
from model import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.package / 'config.json').read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(config['track'], config['model'], config.get('input_normalization', 'none'), config.get('normalization_layer', 'batch')).to(device)
    model.load_state_dict(torch.load(args.package / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    loader = torch.utils.data.DataLoader(Dataset(args.input), batch_size=8, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / 'predictions.jsonl', 'w') as stream, torch.inference_mode():
        for x, _, ids in loader:
            probabilities = model(x.to(device)).softmax(-1).cpu().numpy()
            for sample_id, p in zip(ids, probabilities):
                stream.write(json.dumps({'id': sample_id, 'probabilities': p.tolist()}) + '\n')


if __name__ == '__main__':
    main()
