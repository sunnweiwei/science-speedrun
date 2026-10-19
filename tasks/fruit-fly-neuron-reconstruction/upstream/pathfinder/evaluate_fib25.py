"""Score the official FIB-25 example against mapped GT and its released output.

These are locally derived voxel scores, not a claimed published paper scalar.
All nonzero reference voxels are scored, including unassigned prediction voxels.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from skimage.metrics import adapted_rand_error, variation_of_information
from dataset_contract import sha256


def voxel_metrics(truth, prediction):
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError('Identical 3D grids required; no resizing or cropping')
    if any(a.dtype.kind not in 'iu' or a.min() < 0 for a in (truth, prediction)):
        raise ValueError('Nonnegative integer labels required')
    valid = truth != 0
    if not valid.any():
        raise ValueError('Empty reference')
    # Compact identities to avoid allocating sparse matrices indexed by large IDs.
    # Zero remains zero, and only GT zero is excluded from scoring.
    def compact(a):
        ids, inverse = np.unique(a, return_inverse=True)
        return (inverse + int(ids[0] != 0)).reshape(a.shape)
    gt, pred = compact(truth), compact(prediction)
    split, merge = variation_of_information(gt, pred, ignore_labels=(0,))
    error, precision, recall = adapted_rand_error(gt, pred, ignore_labels=(0,))
    return dict(vi_split_bits=float(split), vi_merge_bits=float(merge),
        vi_total_bits=float(split + merge), adapted_rand_error=float(error),
        adapted_rand_precision=float(precision), adapted_rand_recall=float(recall),
        scored_voxels=int(valid.sum()), omitted_prediction_fraction=float((prediction[valid] == 0).mean()),
        gt_positive_instances=int(np.unique(truth[valid]).size),
        prediction_positive_instances=int(np.unique(prediction[prediction > 0]).size))


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return data['segmentation'].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ground-truth', 'prediction', 'released-prediction', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    if not str(a.output.resolve()).startswith(os.environ.get('PATHFINDER_SCRATCH_ROOT', '/scratch').rstrip('/') + '/') or a.output.exists():
        raise ValueError('New Xtemp output required')
    truth = np.load(a.ground_truth, mmap_mode='r', allow_pickle=False)
    prediction, released = load_prediction(a.prediction), load_prediction(a.released_prediction)
    result = dict(schema='fib25-example-evaluation/v1',
        generated=voxel_metrics(truth, prediction), released=voxel_metrics(truth, released),
        prediction_bytes_equal=bool(np.array_equal(prediction, released)),
        inputs={name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in
            [('mapped_ground_truth',a.ground_truth),('generated_prediction',a.prediction),('released_prediction',a.released_prediction)]},
        adapter_sha256=sha256(__file__), published_scalar_reproduced=False,
        scope='Locally derived full-volume voxel metrics of the official example; GT zero ignored, predicted zero retained; no paper-score claim')
    result['generated_minus_released']={k:result['generated'][k]-result['released'][k]
        for k in ('vi_total_bits','adapted_rand_error','omitted_prediction_fraction')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':main()
