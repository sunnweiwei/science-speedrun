"""Resolve a verified parent-array reference without creating another disk copy."""
import json
from pathlib import Path


def segmentation_path(folder):
    folder=Path(folder);receipt=folder/'receipt.json'
    if not receipt.is_file():return folder/'segmentation.npy'
    r=json.loads(receipt.read_text())
    if 'segmentation_path' not in r:return folder/'segmentation.npy'
    if (r.get('schema')!='prepared-prediction-context/v1' or r.get('storage')!='parent-array-reference'
        or r.get('complete') is not True or r.get('ground_truth_used') is not False):
        raise ValueError('Complete prediction-only reference context required')
    root=Path(r['root']).resolve();parent=Path(r['prediction']).resolve();path=Path(r['segmentation_path'])
    if (not path.is_absolute() or path!=path.resolve() or path!=parent/'segmentation.npy'
        or root not in parent.parents or root not in folder.resolve().parents
        or (folder/'segmentation.npy').exists() or (folder/'segmentation.npy').is_symlink()):
        raise ValueError('Unambiguous parent array inside original root required')
    return path
