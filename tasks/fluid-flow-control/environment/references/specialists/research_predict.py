"""Portable specialist inference for frozen Research v2 native observations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def native_action(action, low, high):
    action = np.asarray(action, dtype=np.float64)
    low, high = np.asarray(low), np.asarray(high)
    if action.shape != low.shape or not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
        raise ValueError('Invalid normalized action')
    return np.clip(low + (np.clip(action, -1, 1) + 1) * .5 * (high - low), low, high)


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / 'config.json').read_text())
    torch.set_num_threads(1)
    model = None
    if not cfg.get('zero_control'):
        if hashlib.sha256((root / 'model.pt').read_bytes()).hexdigest() != cfg['model_sha256']:
            raise ValueError('Model checksum mismatch')
        model = torch.jit.load(str(root / 'model.pt'), map_location='cpu').eval()
    ready = False
    for line in sys.stdin:
        q = json.loads(line)
        if q['protocol'] != 'hydrogym-policy/v1':
            raise ValueError('Unknown protocol')
        reply = {'request_id': q['request_id']}
        if q['op'] == 'reset':
            if q['context']['environment'] != cfg['physics']['environment']:
                raise ValueError('Specialist environment mismatch')
            for key, expected in [('observation_shape', [cfg['observation_size']]),
                                  ('action_shape', [cfg['action_size']]),
                                  ('action_low', cfg['native_action_low']),
                                  ('action_high', cfg['native_action_high'])]:
                if q[key] != expected:
                    raise ValueError(f'Specialist profile mismatch: {key}')
            ready = True
            reply['ready'] = True
        elif q['op'] == 'act':
            obs = torch.tensor(q['observation'], dtype=torch.float32)
            if not ready or list(obs.shape) != [cfg['observation_size']] or not torch.isfinite(obs).all():
                raise ValueError('Invalid observation/reset')
            if model is None:
                action = np.zeros(cfg['action_size'])
            else:
                with torch.inference_mode():
                    normalized = model(obs.unsqueeze(0))[0].numpy()
                action = native_action(normalized, cfg['native_action_low'], cfg['native_action_high'])
            reply['action'] = action.tolist()
        elif q['op'] == 'close':
            reply['closed'] = True
            print(json.dumps(reply, allow_nan=False), flush=True)
            return
        else:
            raise ValueError('Unknown operation')
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
