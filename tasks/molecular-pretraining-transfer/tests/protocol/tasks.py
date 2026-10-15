"""The 24 downstream tasks of the evaluation suite and their metrics (standard library + numpy/scipy only).

Groups: QM9 (Uni-Mol2 Table 4, 6 fine-tunes covering 8 properties), COMPAS-1D (Uni-Mol2, 4 properties),
OpenADMET-ExpansionRx (9 endpoints) and ASAP antiviral-ADMET-2025 (5 endpoints). ADMET targets are stored in the
challenges' scoring space (log10(clip(y, 0) + 1), LogD unchanged), so every metric below is the official one.
"""
from __future__ import annotations

import math

GROUPS = {
    "qm9": ["qm9dft_v2", "qm9dft_v2_alpha", "qm9dft_v2_cv", "qm9dft_v2_mu", "qm9dft_v2_r2", "qm9dft_v2_zpve"],
    "compas1d": ["compas1d_aea", "compas1d_aip", "compas1d_disp", "compas1d_dipmom"],
    "expansionrx": ["exrx_logd", "exrx_ksol", "exrx_hlm_clint", "exrx_mlm_clint", "exrx_caco_2_permeability_papp_a_b",
                    "exrx_caco_2_permeability_efflux", "exrx_mppb", "exrx_mbpb", "exrx_mgmb"],
    "asap": ["asap_hlm", "asap_ksol", "asap_logd", "asap_mdr1_mdckii", "asap_mlm"],
}
TASKS = [t for ts in GROUPS.values() for t in ts]
NUM_TARGETS = {t: 3 if t == "qm9dft_v2" else 1 for t in TASKS}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _kendall_tau_b(a: list[float], b: list[float]) -> float | None:
    try:
        from scipy.stats import kendalltau
    except ImportError:  # O(n^2) fallback, used only by the unit tests on small inputs
        conc = disc = ta = tb = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                da, db = a[i] - a[j], b[i] - b[j]
                if da == 0 and db == 0:
                    continue
                if da == 0:
                    ta += 1
                elif db == 0:
                    tb += 1
                elif da * db > 0:
                    conc += 1
                else:
                    disc += 1
        denom = math.sqrt((conc + disc + ta) * (conc + disc + tb))
        return (conc - disc) / denom if denom else None
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(kendalltau(a, b).statistic)


def target_metrics(pred: list[float], true: list[float]) -> dict:
    """MAE, RAE = MAE / mean|y - mean(y)| (ExpansionRx/ASAP leaderboard), R2, Spearman, Kendall tau-b.
    A constant prediction has no rank correlation: Spearman and Kendall are 0.0 there (the official code reports
    NaN), so every metric is a number."""
    n = len(true)
    mean = sum(true) / n
    mae = sum(abs(p - y) for p, y in zip(pred, true)) / n
    mad = sum(abs(y - mean) for y in true) / n
    ss_res = sum((p - y) ** 2 for p, y in zip(pred, true)); ss_tot = sum((y - mean) ** 2 for y in true)
    return {"mae": mae, "rae": mae / mad, "r2": 1 - ss_res / ss_tot,
            "spearman": _pearson(_ranks(pred), _ranks(true)) or 0.0, "kendall": _kendall_tau_b(pred, true) or 0.0, "n": n}


def task_metrics(pred: dict[str, list[float]], true: dict[str, list[float]]) -> dict:
    """Metrics of one task from per-molecule predictions and labels keyed by SMILES; multi-target tasks
    (qm9dft_v2: HOMO, LUMO, gap) average each metric over targets, as Uni-Mol2's agg_mae does."""
    keys = sorted(true)
    k = len(true[keys[0]])
    per = [target_metrics([pred[s][i] for s in keys], [true[s][i] for s in keys]) for i in range(k)]
    out = {}
    for name in ("mae", "rae", "r2", "spearman", "kendall"):
        out[name] = sum(m[name] for m in per) / k
    out["n"] = len(keys)
    return out


def group_summary(per_task: dict[str, dict]) -> dict:
    """Per group: MA-RAE (mean RAE over the group's tasks, the ExpansionRx/ASAP leaderboard metric) and mean Spearman."""
    return {f"{g}_{name}": sum(per_task[t][key] for t in ts) / len(ts)
            for g, ts in GROUPS.items() for name, key in (("ma_rae", "rae"), ("mean_spearman", "spearman"))}
