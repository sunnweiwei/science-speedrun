"""One trainable Braindecode backbone with five input adapters and heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model import MODELS, SPECS

TRACKS = tuple(SPECS)
CORE_CHANNELS = {'eegnet': 18, 'contrawr': 18, 'cbramod': 18,
                 'biot': 18, 'codebrain': 19}
CHUNK_SAMPLES = 400
PIPELINE_VERSION = 'shared-v2.1-channel-preserving'
FEATURES = {'eegnet': 192, 'contrawr': 256, 'cbramod': 200,
            'biot': 256, 'codebrain': 200}


def chunk_signal(signal):
    """[B,C,T] -> [B*chunks,C,400], preserving channels and chronological time."""
    batch, channels, samples = signal.shape
    n_chunks = math.ceil(samples / CHUNK_SAMPLES)
    padded = F.pad(signal, (0, n_chunks * CHUNK_SAMPLES - samples))
    chunks = padded.reshape(batch, channels, n_chunks, CHUNK_SAMPLES)
    chunks = chunks.permute(0, 2, 1, 3).reshape(batch * n_chunks, channels, CHUNK_SAMPLES)
    weights = signal.new_ones(n_chunks)
    if samples % CHUNK_SAMPLES:
        weights[-1] = (samples % CHUNK_SAMPLES) / CHUNK_SAMPLES
    return chunks, weights


class SharedModel(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        if family not in MODELS:
            raise ValueError(f'unsupported model family: {family}')
        self.family = family
        core_channels = CORE_CHANNELS[family]
        self.backbone = MODELS[family](n_chans=core_channels, n_outputs=2,
                                       n_times=CHUNK_SAMPLES, sfreq=200)
        # The published classifier is not part of the shared representation.
        self.backbone.final_layer = nn.Identity()
        self.adapters = nn.ModuleDict({track: nn.Conv1d(spec[0], core_channels, 1)
                                       for track, spec in SPECS.items()})
        self.heads = nn.ModuleDict({track: nn.Linear(FEATURES[family], spec[2])
                                    for track, spec in SPECS.items()})

    def forward(self, track: str, x: torch.Tensor) -> torch.Tensor:
        if track not in SPECS:
            raise ValueError(f'unknown track: {track}')
        channels, seconds, classes = SPECS[track]
        if x.ndim != 4 or tuple(x.shape[1:]) != (channels, seconds, 200):
            raise ValueError(f'{track}: invalid input shape {tuple(x.shape)}')
        signal = self.adapters[track](x.flatten(2))
        signal, weights = chunk_signal(signal)
        n_chunks = len(weights)
        feature = self.backbone(signal)
        if self.family in ('cbramod', 'codebrain'):
            feature = feature.mean(dim=(1, 2))
        else:
            feature = feature.flatten(1)
        if feature.shape != (x.shape[0] * n_chunks, FEATURES[self.family]):
            raise ValueError(f'{self.family}: unexpected feature shape {tuple(feature.shape)}')
        # Keep every original sample; padded final chunks get proportional weight.
        weights = weights.to(feature.dtype)
        feature = feature.reshape(x.shape[0], n_chunks, -1)
        feature = (feature * weights[None, :, None]).sum(1) / weights.sum()
        logits = self.heads[track](feature)
        if logits.shape != (x.shape[0], classes):
            raise ValueError(f'{track}: invalid output shape {tuple(logits.shape)}')
        return logits
