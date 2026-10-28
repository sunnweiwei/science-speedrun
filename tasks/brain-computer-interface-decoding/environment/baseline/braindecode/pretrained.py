"""Load only audited, pinned backbone tensors; reject partial core loading."""
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file

LOCK = Path(__file__).with_name('pretrained.lock.json')


def load_backbone(model, family, directory):
    if family in ('eegnet', 'contrawr'):
        return {'initialization': 'scratch', 'loaded_core_tensors': 0}
    locked = json.loads(LOCK.read_text())['models'][family]
    path = Path(directory) / f'bci-{family}.safetensors'
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    if hasher.hexdigest() != locked['sha256']:
        raise ValueError(f'{family}: checkpoint checksum mismatch')
    source = load_file(path)
    expected = model.backbone.state_dict()
    missing = sorted(set(expected) - set(source))
    mismatch = sorted(k for k in set(expected) & set(source)
                      if expected[k].shape != source[k].shape)
    extra = sorted(k for k in set(source) - set(expected)
                   if not k.startswith('final_layer.') and k != 'encoder.index')
    if missing or mismatch or extra:
        raise ValueError(f'{family}: incompatible core checkpoint: '
                         f'missing={missing[:5]}, shape={mismatch[:5]}, extra={extra[:5]}')
    model.backbone.load_state_dict({k: source[k] for k in expected}, strict=True)
    return {'initialization': 'pretrained', 'repo': locked['repo'],
            'revision': locked['revision'], 'sha256': locked['sha256'],
            'license': locked['license'], 'source_braindecode': locked['source_braindecode'],
            'loaded_core_tensors': len(expected), 'skipped_classifier_tensors': len(set(source)-set(expected))}
