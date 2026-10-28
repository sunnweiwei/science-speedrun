"""Portable inference for a single five-track model package."""
import argparse
import json
from pathlib import Path

import torch

from data import Dataset
from shared import SharedModel, TRACKS, PIPELINE_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    config = json.loads((Path(__file__).resolve().parent / 'config.json').read_text())
    manifest = json.loads((args.input / 'manifest.json').read_text())
    track = manifest['track']
    if track not in TRACKS:
        raise ValueError('unknown input track')
    if config['schema'] != 'bci-shared-package-v2':
        raise ValueError('unsupported package schema')
    if config.get('pipeline_version') != PIPELINE_VERSION:
        raise ValueError('package preprocessing implementation mismatch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model = SharedModel(config['family']).to(device)
    model.load_state_dict(torch.load(Path(__file__).resolve().parent / 'weights.pt',
                                     map_location=device, weights_only=True), strict=True)
    model.eval()
    loader = torch.utils.data.DataLoader(Dataset(args.input), batch_size=8, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
        for x, _, ids in loader:
            probabilities = model(track, x.to(device)).softmax(-1).cpu().numpy()
            for sample_id, p in zip(ids, probabilities):
                stream.write(json.dumps({'id': sample_id, 'probabilities': p.tolist()}) + '\n')


if __name__ == '__main__':
    main()
