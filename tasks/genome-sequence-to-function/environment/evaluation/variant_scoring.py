"""Variant-effect and perturbation scoring.

Metrics follow the published protocols exactly: AlphaGenome's variant evaluations
(alphagenome_research colabs/variant_eval_examples.ipynb: tissue-weighted AUROC/AUPRC/Spearman
and plain AUPRC) and TraitGym's AUPRC by chromosome, weighted by variants per chromosome.
`published_metric` reproduces AlphaGenome's reported numbers from its released predictions, which
checks this code.

Model scores come from the two predictions of each perturbation (allele A / reference sequence vs
allele B / edited sequence). The perturbation outputs hold, per perturbation and family, `a` and `b`
arrays over the requested slice (dense: [bins, tracks] in the model's asinh space; splice_sites:
[bp, 4]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

AUROC, AUPRC = roc_auc_score, average_precision_score


def spearman(y, p):
    return spearmanr(y, p)[0]


def tissue_weighted(frame: pd.DataFrame, metric, prediction="prediction", target="target") -> float:
    """AlphaGenome's tissue-weighted mean: per-tissue metric weighted by scored rows."""
    values, weights = [], []
    for _, group in frame.groupby("tissue"):
        group = group.dropna(subset=[target, prediction])
        if len(group) < 2:
            continue
        try:
            value = metric(group[target], group[prediction])
        except ValueError:  # single-class tissue
            continue
        if np.isfinite(value):
            values.append(value); weights.append(len(group))
    return float(np.average(values, weights=weights)) if values else float("nan")


def by_chrom_weighted(frame: pd.DataFrame, metric, prediction="prediction", target="label") -> float:
    """TraitGym's main metric: per-chromosome metric weighted by variants per chromosome."""
    values, weights = [], []
    for _, group in frame.groupby("chrom"):
        if group[target].nunique() < 2:
            continue
        values.append(metric(group[target], group[prediction])); weights.append(len(group))
    return float(np.average(values, weights=weights)) if values else float("nan")


# eval name -> (metric, aggregation, published AlphaGenome value on the full release)
METRICS = {
    "eqtl_sign": ("auroc", "tissue", 0.810077),
    "eqtl_causality": ("auroc", "tissue", 0.713255),
    "eqtl_coefficient": ("spearman", "tissue", 0.500588),
    "sqtl_causality": ("auprc", "tissue", 0.7644),
    "clinvar_noncoding": ("auprc", "all", 0.6588),
    "clinvar_splice_region": ("auprc", "all", 0.5699),
    "clinvar_missense": ("auprc", "all", 0.1792),
    "mfass": ("auprc", "all", 0.5120),
    "caqtl": ("auprc", "all", 0.5643),
    "bqtl_spi1": ("auprc", "all", 0.4952),
    "enhancer_gene": ("auprc", "all", 0.7490),
    "traitgym_mendelian": ("auprc", "chrom", None),
    "traitgym_complex": ("auprc", "chrom", None),
}
_FUNCTIONS = {"auroc": AUROC, "auprc": AUPRC, "spearman": spearman}


def metric(name: str, frame: pd.DataFrame, prediction: str = "prediction") -> float:
    kind, aggregation, _ = METRICS[name]
    fn = _FUNCTIONS[kind]
    if aggregation == "tissue":
        return tissue_weighted(frame, fn, prediction)
    if aggregation == "chrom":
        return by_chrom_weighted(frame, fn, prediction)
    clean = frame.dropna(subset=["target", prediction])
    return float(fn(clean["target"], clean[prediction]))


def published_metric(name: str, released: pd.DataFrame) -> float:
    """AlphaGenome's own score from its released prediction table (as in the colab)."""
    frame = released.rename(columns={"alphagenome_prediction": "prediction"})
    if name == "mfass":
        frame = frame.assign(target=frame.target.astype(int))
    return metric(name, frame)


# ----------------------------------------------------------------------------------------------
# Model scores from perturbation outputs (perturb.sh).

import zarr  # noqa: E402

GTEX = "gtex"


def _counts(x):
    """Model outputs are asinh(counts); counts cannot be negative."""
    return np.maximum(np.sinh(np.asarray(x, np.float64)), 0.0)


def _rna_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    return tracks[tracks.family == "RNA_SEQ"].reset_index(drop=True)


def _gtex_ontology(tracks: pd.DataFrame) -> dict[str, str]:
    """GTEx tissue name -> ontology term, read from the (masked) GTEx track names."""
    rna = _rna_tracks(tracks)
    out = {}
    for name, curie in zip(rna.name, rna.ontology_curie):
        parts = str(name).split(" ")
        if len(parts) > 2 and parts[1] == GTEX:
            out[parts[2]] = curie
    return out


def rna_track_columns(tracks: pd.DataFrame, strand: str, ontology: str | None) -> np.ndarray:
    """ENCODE RNA-seq tracks on the gene's strand (or unstranded), restricted to an ontology term
    when that term has ENCODE tracks; GTEx tracks carry no targets in this release."""
    rna = _rna_tracks(tracks)
    keep = (rna.data_source != GTEX) & rna.strand.isin(
        ["POSITIVE" if strand == "+" else "NEGATIVE", "UNSTRANDED"])
    if ontology is not None and (keep & (rna.ontology_curie == ontology)).any():
        keep &= rna.ontology_curie == ontology
    return np.flatnonzero(keep.to_numpy())


def _family_columns(tracks: pd.DataFrame, family: str, ontology: str, name_contains: str | None = None):
    sub = tracks[tracks.family == family].reset_index(drop=True)
    keep = sub.ontology_curie == ontology
    if name_contains:
        keep &= sub.name.str.contains(name_contains, regex=False)
    return np.flatnonzero(keep.to_numpy())


def perturbation_scores(name: str, labels: pd.DataFrame, outputs, index: dict[str, int],
                        tracks: pd.DataFrame) -> pd.Series:
    """One model score per label row, following the matching AlphaGenome / TraitGym scorer."""
    cache: dict = {}

    def arrays(pid, family):
        # Many label rows (genes x tissues) share one perturbation; read its arrays once.
        if (pid, family) not in cache:
            if len(cache) > 64:
                cache.clear()
            group = outputs[f"perturbations/p{index[pid]:06d}"]
            cache[pid, family] = (np.asarray(group[f"a/{family}"], np.float64),
                                  np.asarray(group[f"b/{family}"], np.float64))
        return cache[pid, family]

    labels = labels.sort_values("perturbation_id", kind="stable")
    scores = np.full(len(labels), np.nan)
    if name.startswith(("eqtl", "enhancer")):
        gtex = _gtex_ontology(tracks)
        k562 = "EFO:0002067"
        for i, row in enumerate(labels.itertuples(index=False)):
            a, b = arrays(row.perturbation_id, "rna_seq")
            ontology = gtex.get(getattr(row, "tissue", None)) if name.startswith("eqtl") else k562
            columns = rna_track_columns(tracks, row.gene_strand, ontology)
            bins = np.asarray(row.exon_bins, np.int64)
            mean_a = _counts(a[np.ix_(bins, columns)]).mean(0)
            mean_b = _counts(b[np.ix_(bins, columns)]).mean(0)
            # AlphaGenome's gene_exons_contained_diff_mean with log(x + 0.001) after aggregation.
            delta = float(np.mean(np.log(mean_b + 1e-3) - np.log(mean_a + 1e-3)))
            scores[i] = delta if name in ("eqtl_sign", "eqtl_coefficient") else abs(delta)
    elif name in ("caqtl", "bqtl_spi1"):
        family, columns = (("dnase", _family_columns(tracks, "DNASE", "EFO:0002784")) if name == "caqtl"
                           else ("chip_tf", _family_columns(tracks, "CHIP_TF", "EFO:0002784", "SPI1")))
        for i, row in enumerate(labels.itertuples(index=False)):
            a, b = arrays(row.perturbation_id, family)
            # center_mask_diff_log2_sum (501 bp, here the 128-bp bins it covers), mean |delta| over tracks.
            sa, sb = _counts(a[:, columns]).sum(0), _counts(b[:, columns]).sum(0)
            scores[i] = float(np.mean(np.abs(np.log2(sb + 1) - np.log2(sa + 1))))
    elif name.startswith("traitgym"):
        families = ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]
        for i, row in enumerate(labels.itertuples(index=False)):
            per_track = []
            for family in families:
                a, b = arrays(row.perturbation_id, family)
                per_track.append(np.sqrt(((b - a) ** 2).sum(0)))  # L2 over positions, per track
            scores[i] = float(np.sqrt((np.concatenate(per_track) ** 2).sum()))  # then L2 over tracks
    else:  # splicing: largest change in splice-site probability near the variant
        for i, row in enumerate(labels.itertuples(index=False)):
            a, b = arrays(row.perturbation_id, "splice_sites")
            sigmoid = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
            scores[i] = float(np.abs(sigmoid(b) - sigmoid(a)).max())
    return pd.Series(scores, index=labels.index)


def score_perturbations(perturbation_root: Path, labels_root: Path) -> dict:
    """All variant/perturbation metrics, with the anchors computed on the same rows."""
    outputs = zarr.open_group(str(perturbation_root), mode="r")
    ids = list(outputs.attrs["perturbation_ids"])
    index = {pid: i for i, pid in enumerate(ids)}
    tracks = pd.read_parquet(labels_root / "labels/tracks.parquet")
    report = {}
    for name in METRICS:
        labels = pd.read_parquet(labels_root / f"labels/{name}.parquet")
        model = perturbation_scores(name, labels, outputs, index, tracks)
        frame = labels.assign(prediction=model)
        if name == "mfass":
            frame["target"] = frame.target.astype(int)
        entry = {"rows": len(frame), "perturbations": int(frame.perturbation_id.nunique()),
                 "metric": METRICS[name][0], "model": metric(name, frame)}
        if name.startswith("traitgym"):
            entry["anchors"] = {c[len("anchor_"):]: metric(name, frame, prediction=c)
                                for c in frame.columns if c.startswith("anchor_")}
        else:
            entry["anchors"] = {"AlphaGenome": metric(name, frame, prediction="alphagenome_prediction")}
        report[name] = entry
    return report


PERTURBATION_SCHEMA = "genome-sequence-to-function-perturbations/v1"
_DENSE_TRACK_COUNTS = {"atac": 167, "dnase": 305, "procap": 12, "cage": 546, "rna_seq": 667,
                       "chip_tf": 1617, "chip_histone": 1116}


def validate_perturbations(root: Path, perturbations: pd.DataFrame) -> None:
    """Every requested slice, for both sequences, with the right shape and finite values."""
    group = zarr.open_group(str(root), mode="r")
    if group.attrs.get("schema") != PERTURBATION_SCHEMA:
        raise ValueError("perturbation schema mismatch")
    if list(group.attrs.get("perturbation_ids", [])) != perturbations.perturbation_id.astype(str).tolist():
        raise ValueError("perturbation ID order mismatch")
    for index, row in enumerate(perturbations.itertuples(index=False)):
        item = group[f"perturbations/p{index:06d}"]
        for family in row.families:
            shape = ((row.splice_end - row.splice_start, 4) if family == "splice_sites"
                     else (row.dense_bin_end - row.dense_bin_start, _DENSE_TRACK_COUNTS[family]))
            for side in "ab":
                array = item[f"{side}/{family}"]
                if tuple(array.shape) != shape or array.dtype not in (np.dtype("float16"), np.dtype("float32")):
                    raise ValueError(f"perturbation {index} {side}/{family} has {array.shape}; expected {shape}")
                if not np.isfinite(array[:]).all():
                    raise ValueError(f"perturbation {index} {side}/{family} contains non-finite values")
