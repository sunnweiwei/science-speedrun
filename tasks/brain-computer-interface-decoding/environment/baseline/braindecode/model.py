"""Braindecode 1.8.1 classifiers trained exclusively from random weights."""
import torch
from torch import nn
from braindecode.models import EEGNet, ContraWR, CBraMod, BIOT, CodeBrain

SPECS = {'faced': (32, 10, 9), 'tuev': (16, 5, 6), 'ad65': (19, 10, 3),
         'openbmi': (62, 4, 2), 'longitudinal': (58, 2, 2)}
MODELS = {'eegnet': EEGNet, 'contrawr': ContraWR, 'cbramod': CBraMod,
          'biot': BIOT, 'codebrain': CodeBrain}


def replace_batchnorm(module):
    """Use one-group GroupNorm, preserving each BatchNorm affine initialization."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            replacement = nn.GroupNorm(1, child.num_features, eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm(child)


class Model(nn.Module):
    def __init__(self, track: str, family: str, input_normalization: str = "none", normalization_layer: str = "batch"):
        super().__init__()
        if track not in SPECS or family not in MODELS:
            raise ValueError(f'unsupported track/model: {track}/{family}')
        if input_normalization not in ('none','per-channel-zscore'):
            raise ValueError('unsupported input normalization')
        if normalization_layer not in ('batch', 'group'):
            raise ValueError('unsupported normalization layer')
        self.normalization_layer = normalization_layer
        self.input_normalization = input_normalization
        chans, seconds, classes = SPECS[track]
        self.net = MODELS[family](n_chans=chans, n_outputs=classes,
                                  n_times=seconds * 200, sfreq=200)
        if normalization_layer == 'group':
            replace_batchnorm(self.net)
        self.track, self.family = track, family

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:4] != (SPECS[self.track][0], SPECS[self.track][1], 200):
            raise ValueError(f'invalid input shape: {tuple(x.shape)}')
        signal = x.flatten(start_dim=2)
        if self.input_normalization == 'per-channel-zscore':
            signal = (signal - signal.mean(-1, keepdim=True)) / signal.std(
                -1, keepdim=True, correction=0).clamp_min(1e-6)
        logits = self.net(signal)
        if logits.shape != (x.shape[0], SPECS[self.track][2]):
            raise ValueError(f'invalid output shape: {tuple(logits.shape)}')
        return logits
