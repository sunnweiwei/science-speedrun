"""Author-documented two-seed FFN resegmentation score and merge rule.

Source: google/ffn doc/manual.md, Agglomeration through resegmentation, pinned
b8df2d96d7c8da057fd2702ced19732a08ab6ae4. These pure scoring functions do not
run the model or choose decision points. Callers must provide two independent
seed-and-grow results, evaluated in the same analysis window. No GT is used.
"""
import numpy as np

AUTHOR_COMMIT = 'b8df2d96d7c8da057fd2702ced19732a08ab6ae4'


def score_pair(base, pair, from_a, from_b, deleted_a, deleted_b):
    base = np.asarray(base)
    a, b = np.asarray(from_a), np.asarray(from_b)
    if base.ndim != 3 or base.dtype.kind not in 'iu':
        raise ValueError('Require integer 3D base segmentation')
    if a.shape != base.shape or b.shape != base.shape or a.dtype != bool or b.dtype != bool:
        raise ValueError('Require two boolean masks in identical analysis coordinates')
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0 or pair[0] == pair[1]:
        raise ValueError('Two distinct positive segment IDs required')
    original = [base == sid for sid in pair]
    if not all(x.any() for x in original):
        raise ValueError('Both original fragments must occur in analysis window')
    for deleted in (deleted_a, deleted_b):
        if type(deleted) != int or deleted < 0:
            raise ValueError('Deletion counts must be nonnegative integers')
    union = int((a | b).sum())
    return dict(pair=list(map(int, pair)), iou=float((a & b).sum()/union) if union else 0.,
        consistency=[[float((prediction & fragment).sum()/fragment.sum()) for fragment in original]
                     for prediction in (a, b)],
        prediction_voxels=[int(a.sum()), int(b.sum())],
        deleted_voxels=[deleted_a, deleted_b], author_commit=AUTHOR_COMMIT)


def accept_pair(score, *, complete, iou_threshold=.8, consistency_threshold=.6,
                deletion_fraction_threshold=.02):
    """Conservative manual rule; empty/incomplete resegmentations never pass.

    The deletion condition is OR across seeds; consistency is AND across all
    four seed/original-fragment combinations. Thresholds are strict, as in the
    author's manual. Any calibration belongs on validation data only.
    """
    if type(complete) != bool:
        raise ValueError('Explicit completion flag required')
    thresholds = (iou_threshold, consistency_threshold, deletion_fraction_threshold)
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in thresholds):
        raise ValueError('Thresholds must be finite fractions')
    if not complete:
        return False
    consistency = np.asarray(score['consistency'])
    sizes = np.asarray(score['prediction_voxels'])
    deletions = np.asarray(score['deleted_voxels'])
    if consistency.shape != (2, 2) or sizes.shape != (2,) or deletions.shape != (2,):
        raise ValueError('Invalid score shape')
    if not np.isfinite(consistency).all() or not np.isfinite(score['iou']):
        raise ValueError('Nonfinite score')
    if (sizes <= 0).any():
        return False
    return bool(score['iou'] > iou_threshold and
                (consistency > consistency_threshold).all() and
                (deletions/sizes < deletion_fraction_threshold).any())


def accept_pathfinder_pair(score, *, complete):
    """PATHFINDER v1 supplementary FFN-segmentation gate, distinct from manual.

    ((min deletion fraction < .02) AND IoU > .8 AND min consistency > .8)
    OR (IoU > .9 AND min consistency > .9). Multi-checkpoint acceptance is
    a separate AND over checkpoint-specific decisions. No GT is consulted.
    Source: doi:10.1101/2025.05.16.654254, supplementary FFN segmentation.
    """
    if type(complete) is not bool:raise ValueError('Explicit completion flag required')
    if not complete:return False
    consistency=np.asarray(score['consistency'],dtype=float)
    sizes=np.asarray(score['prediction_voxels'],dtype=float)
    deleted=np.asarray(score['deleted_voxels'],dtype=float)
    iou=float(score['iou'])
    if (consistency.shape!=(2,2) or sizes.shape!=(2,) or deleted.shape!=(2,) or
            not np.isfinite(consistency).all() or not np.isfinite(sizes).all() or
            not np.isfinite(deleted).all() or not np.isfinite(iou) or
            not 0<=iou<=1 or (consistency<0).any() or (consistency>1).any() or
            (sizes<0).any() or (deleted<0).any() or
            (sizes!=np.floor(sizes)).any() or (deleted!=np.floor(deleted)).any()):
        raise ValueError('Invalid resegmentation scores')
    if (sizes==0).any():return False
    return bool(((deleted/sizes<.02).any() and iou>.8 and (consistency>.8).all()) or
                (iou>.9 and (consistency>.9).all()))
