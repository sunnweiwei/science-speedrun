"""Structural submission checks; dataset/runtime failures remain infrastructure errors."""
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np


class InvalidSubmission(ValueError):
    pass


def validate_package(model):
    model = Path(model)
    for name in ('encoder.pt', 'config.json', 'infer.py', 'provenance.json', 'LICENSE-OlmoEarth'):
        path = model / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise InvalidSubmission(f'Missing nonempty regular package file: {name}')
    for name in ('config.json', 'provenance.json'):
        try:
            value = json.loads((model / name).read_text())
        except (ValueError, UnicodeError) as error:
            raise InvalidSubmission(f'Invalid {name}') from error
        if not isinstance(value, dict):
            raise InvalidSubmission(f'{name} must contain an object')
        if name == 'provenance.json':
            if value.get('training_validity') == 'invalid':
                raise InvalidSubmission('Known invalid training implementation')
            if value.get('origin') == 'official_public_pretrained_reference':
                raise InvalidSubmission('Public pretrained reference is ineligible')


def validate_features(path, inputs):
    # Read trusted input outside the candidate-error handler.
    with h5py.File(inputs, 'r') as source:
        expected = source['ids'].asstr()[:].tolist()
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise InvalidSubmission('Missing regular prediction file')
    try:
        with np.load(path, allow_pickle=False) as data:
            ids, features = data['ids'], data['features']
            if (ids.ndim != 1 or ids.dtype.kind not in 'US'
                    or len(ids) != len(expected) or len(set(ids.tolist())) != len(ids)
                    or set(ids.tolist()) != set(expected)):
                raise InvalidSubmission('IDs missing, duplicate or extra')
            if (features.ndim != 2 or features.shape[0] != len(expected)
                    or features.shape[1] == 0 or features.dtype != np.float32
                    or not np.isfinite(features).all()):
                raise InvalidSubmission('Expected finite float32 features [N,D]')
    except (ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise InvalidSubmission(str(error)) from error
