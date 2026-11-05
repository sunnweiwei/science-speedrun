"""Public ranking metrics, without private data readers or verifier orchestration."""

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC, CalcEnrichment

FRACTIONS = (0.005, 0.01, 0.02, 0.05)


def ranking_metrics(values) -> dict:
    """RDKit metrics, averaging all possible orderings within exact score ties."""
    scores = np.asarray(values, dtype=np.float64)
    count = len(scores)
    if len(np.unique(scores[:, 0])) == count:
        ef = CalcEnrichment(values, 1, FRACTIONS)
        return {"roc_auc": float(CalcAUC(values, 1)),
                "bedroc_80.5": float(CalcBEDROC(values, 1, 80.5)),
                **{f"ef_{f:g}": float(v) for f, v in zip(FRACTIONS, ef, strict=True)}}
    # Sorting by labels or retaining an actives-first LMDB order would let a
    # constant predictor obtain perfect enrichment. Use expected tie ranks.
    labels = scores[:, 1]
    active = int(labels.sum())
    expected = np.empty(count)
    rank_sum = 0.0
    boundaries = np.r_[0, np.flatnonzero(np.diff(scores[:, 0])) + 1, count]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        positives = labels[start:end].sum()
        expected[start:end] = positives / (end - start)
        rank_sum += positives * (start + 1 + end) / 2
    auc = 1 - (rank_sum - active * (active + 1) / 2) / (active * (count - active))
    weights = np.exp(-80.5 * np.arange(1, count + 1) / count)
    maximum, minimum = weights[:active].sum(), weights[-active:].sum()
    bedroc = (np.dot(expected, weights) - minimum) / (maximum - minimum)
    prefix = np.cumsum(expected)
    return {"roc_auc": float(auc), "bedroc_80.5": float(np.clip(bedroc, 0, 1)),
            **{f"ef_{f:g}": float(prefix[int(np.ceil(count * f)) - 1] * count /
                                 (int(np.ceil(count * f)) * active)) for f in FRACTIONS}}

