"""Two-seed FFN resegmentation using the shared local FFN15 predictor.

Independent NumPy/PyTorch adaptation of google/ffn's documented algorithm.
Retains this track's FFN movement/assignment semantics, not author bit parity.
The predictor receives a batch of image/logit windows and returns logits.
"""
from ffn_movement import FaceMaxMovement,disconnected_bias
import numpy as np
from exact_fragment_edt import fragment_distance
from ffn_agglomeration import score_pair, accept_pair


def grow(image, seed, predict, *, max_calls, analysis_radius=34, occupied=None):
    """One seed; deletion history counts confident-positive to negative flips."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or min(image.shape) < 33:
        raise ValueError('Require uint8 3D EM with 33-cube real context')
    if occupied is not None and (occupied.shape!=image.shape or occupied.dtype!=np.bool_):
        raise ValueError('Aligned boolean occupied mask required')
    seed = np.asarray(seed, dtype=int)
    if (seed < 16).any() or (seed >= np.array(image.shape)-16).any() or max_calls < 1:
        raise ValueError('Seed needs full context and a positive call limit')
    canvas = np.full(image.shape, -2.944438979, dtype=np.float32)
    canvas[tuple(seed)] = 2.944438979
    initialized=np.zeros(canvas.shape,bool);initialized[tuple(seed)]=True
    movement=FaceMaxMovement(seed,canvas.shape); calls = deleted = 0
    exhausted=False; moves=dict(seeds_lost=0,stale_moves_skipped=0)
    center = np.array(image.shape)//2
    while movement.queue:
        point = movement.next(canvas,moves,blocked=None if occupied is None else lambda point: occupied[point])
        if point is None:break
        if calls>=max_calls:
            exhausted=True;break
        window = tuple(slice(c-16,c+17) for c in point)
        old = canvas[window].copy()
        logits = np.asarray(predict(np.array(image[window],np.float32)[None]/255, old[None]))
        if logits.shape != (1,33,33,33) or not np.isfinite(logits).all():
            raise ValueError('Predictor must return finite Bx33x33x33 logits')
        logits = logits[0]
        if (np.abs(np.array(point)-center) <= analysis_radius).all():
            deleted += int(((old >= np.log(4.)) & (logits < 0)).sum())
        logits=disconnected_bias(old,logits,initialized[window],audit=moves)
        canvas[window] = logits; initialized[window]=True; calls += 1
        movement.update(logits,point)
    # Only a still-valid unprocessed move makes a call cap incomplete.
    return canvas >= 2.944439, dict(calls=calls, deleted_voxels=deleted, complete=not exhausted, **moves)


def resegment_pair(image, base, pair, predict, *, retries=8, exclusion_radius=8,
                   recovery=.6, analysis_radius=34, max_calls_per_seed=4096):
    image, base = np.asarray(image), np.asarray(base)
    if image.shape != base.shape or image.shape != (101,)*3:
        raise ValueError('Require aligned 101-cube raw/base around decision point')
    if len(pair) != 2 or pair[0] == pair[1] or min(pair) <= 0:
        raise ValueError('Distinct positive pair required')
    if retries < 1 or exclusion_radius < 0 or not 0 <= recovery <= 1 or not 0 < analysis_radius <= 34:
        raise ValueError('Invalid resegmentation settings')
    center = 50
    analysis = (slice(center-analysis_radius,center+analysis_radius+1),)*3
    occupied=(base>0)&(base!=pair[0])&(base!=pair[1])
    masks = []; reports = []
    for sid in pair:
        original = base == sid
        if not original[analysis].any():
            return dict(pair=list(pair), accepted=False, complete=False, reason='fragment_absent_in_analysis')
        distance = fragment_distance(original)
        for axis in range(3):
            sl = [slice(None)]*3; sl[axis] = slice(0,16); distance[tuple(sl)] = 0
            sl[axis] = slice(-16,None); distance[tuple(sl)] = 0
        attempts = []; recovered = False; mask = None
        for _ in range(retries):
            seed = np.array(np.unravel_index(np.argmax(distance), distance.shape))
            if distance[tuple(seed)] <= 0:
                break
            excluded = tuple(slice(max(0,c-exclusion_radius), min(n,c+exclusion_radius+1)) for c,n in zip(seed,distance.shape))
            distance[excluded] = 0
            mask, report = grow(image, seed, predict, max_calls=max_calls_per_seed,
                                analysis_radius=analysis_radius,occupied=occupied)
            report['seed_zyx'] = seed.tolist()
            report['recovery'] = float((mask[analysis] & original[analysis]).sum()/original[analysis].sum())
            attempts.append(report)
            if report['complete'] and report['recovery'] >= recovery:
                recovered = True; break
            if not report['complete']:
                break
        if mask is None:
            return dict(pair=list(pair), accepted=False, complete=False, reason='no_context_safe_seed')
        masks.append(mask[analysis]); reports.append(dict(attempts=attempts, recovered=recovered))
    scores = score_pair(base[analysis], pair, *masks,
                        reports[0]['attempts'][-1]['deleted_voxels'],
                        reports[1]['attempts'][-1]['deleted_voxels'])
    complete = all(r['attempts'][-1]['complete'] for r in reports)
    return dict(**scores, complete=complete, attempts=reports,
                accepted=accept_pair(scores, complete=complete),
                adaptation='shared local FFN15 grow semantics; author manual resegmentation gates; no bit-parity claim')
