"""Revalidate FFN moves against the current canvas, not stale queued scores.

Matches the safety checks in pinned Google FFN FaceMaxMovementPolicy.__next__
and Canvas.segment_at: reject retracted positions and stop if the initial seed
falls below move_threshold. The local candidate-generation policy is unchanged.
"""
MOVE_THRESHOLD = 2.197224577


def next_move(queue, visited, canvas, seed, audit=None):
    if canvas[tuple(seed)] < MOVE_THRESHOLD:
        if audit is not None:audit['seeds_lost'] = audit.get('seeds_lost',0)+1
        queue.clear()
        return None
    while queue:
        point=queue.popleft();key=tuple(point)
        if key in visited:continue
        if canvas[key] < MOVE_THRESHOLD:
            if audit is not None:audit['stale_moves_skipped']=audit.get('stale_moves_skipped',0)+1
            continue
        return point
    return None


def disconnected_bias(old_logits, predicted, initialized, threshold=0., audit=None):
    """Google Canvas.update_at disconnectedness bias (proto default 0).

    The initialized mask is essential: padding is unknown, not a previously
    predicted disconnection. Positive-to-negative corrections remain allowed.
    """
    import numpy as np
    if not np.isfinite(threshold) or threshold>1:raise ValueError('Finite threshold <=1 required; negative disables bias')
    if old_logits.shape!=predicted.shape or initialized.shape!=predicted.shape or initialized.dtype!=np.bool_:raise ValueError('Matching logits and boolean initialized mask required')
    result=predicted.copy()
    if threshold>=0 and np.mean(result>=MOVE_THRESHOLD)>threshold:
        frozen=initialized & (old_logits<0) & (result>old_logits)
        result[frozen]=old_logits[frozen]
        if audit is not None:audit['disconnected_voxel_updates_blocked']=audit.get('disconnected_voxel_updates_blocked',0)+int(frozen.sum())
    return result


def face_max_offsets(logits, deltas=(8,8,8), threshold=MOVE_THRESHOLD):
    """Face maxima in ZYX; equivalent to Google's get_scored_move_offsets."""
    import numpy as np
    center=np.array(logits.shape)//2
    if logits.ndim!=3 or len(deltas)!=3 or any(d<0 or d>c for d,c in zip(deltas,center)):
        raise ValueError('Three nonnegative deltas within the prediction window required')
    slices=[slice(c-d,c+d+1) for c,d in zip(center,deltas)];found=set()
    for axis,delta in enumerate(deltas):
        if delta==0:continue
        for offset in (-delta,delta):
            face_slices=slices.copy();face_slices[axis]=int(center[axis]+offset)
            face=logits[tuple(face_slices)];point=np.unravel_index(face.argmax(),face.shape);score=float(face[point])
            if score<threshold:continue
            relative=[int(p-n//2) for p,n in zip(point,face.shape)];relative.insert(axis,int(offset))
            found.add((score,tuple(relative)))
    return sorted(found,reverse=True)


class FaceMaxMovement:
    """FIFO face proposals with author-style seed-relative visited cells."""
    def __init__(self,seed,shape,margin=16,deltas=(8,8,8)):
        from collections import deque
        self.seed=tuple(map(int,seed));self.shape=tuple(shape);self.margin=margin;self.deltas=tuple(deltas)
        self.queue=deque([self.seed]);self.visited=set()

    def key(self,point):
        return tuple((int(p)-s+d//2)//max(d,1) for p,s,d in zip(point,self.seed,self.deltas))

    def next(self,canvas,audit=None,blocked=None):
        if canvas[self.seed]<MOVE_THRESHOLD:
            if audit is not None:audit['seeds_lost']=audit.get('seeds_lost',0)+1
            self.queue.clear();return None
        while self.queue:
            point=self.queue.popleft()
            if self.key(point) in self.visited:continue
            if canvas[point]<MOVE_THRESHOLD:
                if audit is not None:audit['stale_moves_skipped']=audit.get('stale_moves_skipped',0)+1
                continue
            if blocked is not None and blocked(point):
                if audit is not None:audit['occupied_moves_skipped']=audit.get('occupied_moves_skipped',0)+1
                continue
            return point
        return None

    def update(self,predicted,point):
        self.visited.add(self.key(point))
        for _,offset in face_max_offsets(predicted,self.deltas):
            candidate=tuple(int(p)+d for p,d in zip(point,offset))
            if all(self.margin<=p<n-self.margin for p,n in zip(candidate,self.shape)):
                self.queue.append(candidate)
