"""Two-pass per-request query-independence verification gate.

Under the V4 contract (`instruction.md` and `environment/contract.md`), submitted models must
predict each perturbation request independently of batch composition, request ordering, or other
targets in `requests.jsonl`, seeding any stochastic sampling deterministically per `request_id`
(or the request's `seed`).

The untrusted `main` verifier executes:
  - Pass 1: full `requests.jsonl` manifest -> `predictions.h5ad`
  - Pass 2: deterministic shuffled proper subset of `requests.jsonl` in a fresh process ->
            `predictions_subset.h5ad`

The trusted `grader.py` compares the per-request mean `log1p(CP10k)` expression profiles across
both passes via `compare_request_predictions` (`atol=1e-4, rtol=1e-4`) and rejects batch-coupled
or cross-query shrinkage submissions (`score = reward = 0.0`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4
SUBSET_SEED = 20261115
SUBSET_FRACTION = 0.25
MIN_SUBSET_REQUESTS = 4
MAX_SUBSET_REQUESTS = 64


def mean_log_profile(matrix):
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError('Invalid counts')
    total = np.asarray(x.sum(axis=1)).ravel()
    if not np.isfinite(total).all():
        raise ValueError('Nonfinite row totals')
    x = sparse.diags(1e4 / np.where(total == 0, 1.0, total)) @ x
    x.data = np.log1p(x.data)
    return np.asarray(x.mean(axis=0)).ravel()


def compare_request_predictions(full, subset, *, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
    """Maps request IDs to count matrices in a trusted, identical gene order."""
    if not subset or not set(subset) <= set(full):
        raise ValueError('Invalid common request set')
    rows = {}
    for key, value in subset.items():
        if full[key].shape != value.shape:
            raise ValueError('Cell count or gene panel changed')
        a, b = mean_log_profile(full[key]), mean_log_profile(value)
        difference = np.abs(a - b)
        rows[key] = {
            'max_absolute_logmean_difference': float(difference.max()),
            'within_numerical_tolerance': bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        }
    changed = sum(not r['within_numerical_tolerance'] for r in rows.values())
    max_diff = max((r['max_absolute_logmean_difference'] for r in rows.values()), default=0.0)
    return {
        'status': 'passed' if changed == 0 else 'failed',
        'passed': changed == 0,
        'atol': float(atol),
        'rtol': float(rtol),
        'checked_requests': len(rows),
        'changed_requests': int(changed),
        'max_difference': float(max_diff),
        'requests': rows,
    }


def select_probe_subset(
    requests: list[dict],
    *,
    context_name: str = '',
    seed: int = SUBSET_SEED,
    fraction: float = SUBSET_FRACTION,
    max_requests: int = MAX_SUBSET_REQUESTS,
) -> list[dict]:
    """Deterministically select a shuffled proper subset of `requests` for Pass 2."""
    n = len(requests)
    if n == 0:
        raise ValueError('Cannot select subset from empty request list')
    if n == 1:
        return [dict(requests[0])]
    desired = int(round(n * fraction))
    k = min(n - 1, min(max_requests, max(min(MIN_SUBSET_REQUESTS, n - 1), desired)))
    digest = hashlib.sha256(f'vcz-qi-v4:{seed}:{context_name}:{n}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], 'big'))
    indices = rng.choice(n, size=k, replace=False)
    return [dict(requests[int(i)]) for i in indices]


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate request_id in {path}')
    return rows


def _load_request_matrices(
    predictions_path: Path,
    expected_genes: list[str],
    expected_requests: list[dict],
    keep_ids: set[str],
) -> dict[str, sparse.csr_matrix]:
    import anndata as ad
    from prediction_safety import validate_prediction_storage

    validate_prediction_storage(predictions_path)
    pred = ad.read_h5ad(predictions_path)
    if pred.var_names.has_duplicates or list(map(str, pred.var_names)) != expected_genes:
        raise ValueError('prediction genes/order differ from basal input')
    if 'request_id' not in pred.obs:
        raise ValueError('predictions need obs.request_id')
    ids = pred.obs['request_id'].astype(str).to_numpy()
    unique_ids, counts = np.unique(ids, return_counts=True)
    expected_counts = {r['request_id']: int(r['n_cells']) for r in expected_requests}
    if dict(zip(unique_ids.tolist(), counts.tolist())) != expected_counts:
        raise ValueError('prediction request IDs or per-request cell counts differ from requests.jsonl')
    mat = sparse.csr_matrix(pred.X, dtype=np.float64)
    if not np.isfinite(mat.data).all() or (mat.data < 0).any():
        raise ValueError('predictions contain negative or non-finite values')
    out: dict[str, sparse.csr_matrix] = {}
    for rid in keep_ids:
        mask = ids == rid
        out[rid] = mat[mask]
    return out


def compare_prediction_files(
    full_h5ad: Path,
    subset_h5ad: Path,
    basal_h5ad: Path,
    full_requests_jsonl: Path,
    subset_requests_jsonl: Path,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict:
    """Validate full and subset `.h5ad` outputs and compare common request profiles."""
    import anndata as ad

    basal = ad.read_h5ad(basal_h5ad, backed='r')
    try:
        genes = list(map(str, basal.var_names))
    finally:
        basal.file.close()
    full_reqs = _read_jsonl(full_requests_jsonl)
    subset_reqs = _read_jsonl(subset_requests_jsonl)
    if not subset_reqs:
        raise ValueError('Empty subset request manifest')
    full_by_id = {r['request_id']: r for r in full_reqs}
    for req in subset_reqs:
        rid = req['request_id']
        if rid not in full_by_id or full_by_id[rid] != req:
            raise ValueError(f'Subset request {rid!r} does not match full request manifest')
    if len(full_reqs) >= 2 and len(subset_reqs) >= len(full_reqs):
        raise ValueError('Subset pass must use a strict subset when >= 2 requests exist')
    keep_ids = {r['request_id'] for r in subset_reqs}
    full_map = _load_request_matrices(full_h5ad, genes, full_reqs, keep_ids)
    subset_map = _load_request_matrices(subset_h5ad, genes, subset_reqs, keep_ids)
    return compare_request_predictions(full_map, subset_map, atol=atol, rtol=rtol)
