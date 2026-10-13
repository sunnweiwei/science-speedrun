#!/usr/bin/env python3
"""Build the variant-effect and perturbation evaluation (verifier-only labels, public queries).

    python build_variant_evals.py --sources EVAL_SOURCES --release RELEASE --output OUT

EVAL_SOURCES holds the published evaluation tables, fetched once by fetch_variant_evals.sh:
  ag-evals/<name>.feather   AlphaGenome's 11 variant evaluations (variants, targets and AlphaGenome's
                            own scores; storage.googleapis.com/alphagenome/evals, CC-BY 4.0)
  traitgym/*.parquet        TraitGym causal-variant sets and Borzoi/Enformer/CADD/GPN-MSA scores
                            (songlab/TraitGym @ 1fde195, MIT)
  gencode.v46.annotation.gtf.gz.feather   gene and exon coordinates (AlphaGenome's mirror)

Every query is one *perturbation*: a 1,048,576-bp window and two sequences that differ by an edit
(allele A vs allele B at one base, or the reference vs an N-masked interval). The model predicts both
sequences and returns the requested output slices; the verifier turns them into scores
(environment/evaluation/variant_scoring.py). Only perturbations whose edited base lies in a held-out
test-fold target region (fold1) are used for the AlphaGenome evaluations, so no evaluated locus had
training targets (TraitGym is the documented exception, see below). Each evaluation
is a seeded, label-stratified subset; AlphaGenome's and TraitGym's published per-variant scores for
exactly these rows are stored as anchors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_BP = 1_048_576
TARGET_BP = 196_608
TARGET_OFFSET = (INPUT_BP - TARGET_BP) // 2
BIN = 128
SEED = 20261001
SPLICE_HALF_WINDOW = 2_500     # bp each side of the variant read from splice_sites
CENTER_HALF_WIDTH = 250        # AlphaGenome's center_mask width=501
TRAITGYM_HALF_WINDOW = 1_024   # bp each side read from every dense family for TraitGym
TRAITGYM_MATCH_GROUPS = 80     # one causal variant + nine matched controls per group

EVALS = {
    # name: (source feather, kind, per-eval row budget, family)
    "eqtl_sign": ("eqtl_variant_borzoi_sign_human", "eqtl", 500, "rna_seq"),
    "eqtl_coefficient": ("eqtl_variant_borzoi_coefficient_human", "eqtl", 500, "rna_seq"),
    "eqtl_causality": ("eqtl_variant_catalogue_causality_gene_balanced_human", "eqtl", 500, "rna_seq"),
    "sqtl_causality": ("sqtl_variant_causality_gene_human", "splice", 500, "splice_sites"),
    "clinvar_noncoding": ("clinvar_noncoding", "splice", 500, "splice_sites"),
    "clinvar_splice_region": ("clinvar_splice_site_region", "splice", 500, "splice_sites"),
    "clinvar_missense": ("clinvar_missense", "splice", 500, "splice_sites"),
    "mfass": ("mfass_splicing", "splice", 500, "splice_sites"),
    "caqtl": ("caqtl_african_variant_causality_human", "center", 500, "dnase"),
    "bqtl_spi1": ("bqtl_spi1_variant_causality_human", "center", 500, "chip_tf"),
    "enhancer_gene": ("enhancer_gene_linking_e2g", "enhancer", 500, "rna_seq"),
}


def stable_hash(text: str) -> float:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:12], 16) / 16**12


def fold_regions(bed: Path, fold: str) -> dict[str, np.ndarray]:
    regions: dict[str, list] = {}
    with gzip.open(bed, "rt") as stream:
        for line in stream:
            chrom, start, end, name = line.split()[:4]
            if name == fold:
                regions.setdefault(chrom, []).append((int(start), int(end)))
    return {c: np.array(sorted(v)) for c, v in regions.items()}


def in_fold(regions, chrom: str, pos0: int) -> bool:
    spans = regions.get(chrom)
    if spans is None:
        return False
    i = np.searchsorted(spans[:, 0], pos0, side="right") - 1
    return bool(i >= 0 and spans[i, 0] <= pos0 < spans[i, 1])


def parse_variant(variant_id: str):
    chrom, pos, a, b = variant_id.split("_")[:4]
    return chrom, int(pos) - 1, a, b  # 0-based


# Binary evaluations whose positives are rare at natural prevalence (ClinVar about 2 %, bQTL 0.7 %)
# use a fixed design instead, so the subset carries enough positives: POSITIVES positives and
# NEGATIVES_PER_POSITIVE matched-size negatives drawn at random. Anchors are scored on the same rows.
POSITIVES = 100
NEGATIVES_PER_POSITIVE = 4
FIXED_DESIGN = {"clinvar_noncoding", "clinvar_splice_region", "clinvar_missense", "mfass", "caqtl",
                "bqtl_spi1", "enhancer_gene"}
# bQTL has 18 positives in fold1 (138 genome-wide), so it is drawn genome-wide like TraitGym.
GENOME_WIDE = {"bqtl_spi1"}

# The development split, with labels visible to the agent: the same evaluations at about 40 % of
# the test size, drawn from AlphaGenome's validation fold (fold0) and disjoint from the test rows.
# Genome-wide evaluations exclude the test variants / match groups. When fold0 holds fewer than
# VALID_POSITIVES positives, the pool widens to every locus outside the test fold (fold1).
SPLITS = {
    "test": dict(fold="fold1", positives=POSITIVES, budget=None, match_groups=TRAITGYM_MATCH_GROUPS),
    "valid": dict(fold="fold0", positives=40, budget=200, match_groups=30),
}
VALID_POSITIVES = SPLITS["valid"]["positives"]


def fixed_design(frame: pd.DataFrame, key: str, label: pd.Series, positives_budget: int = POSITIVES) -> list:
    per_key = frame.assign(_l=label).groupby(key)["_l"].max()
    positives = sorted(per_key.index[per_key], key=stable_hash)[:positives_budget]
    negatives = sorted(per_key.index[~per_key], key=stable_hash)[:NEGATIVES_PER_POSITIVE * len(positives)]
    return positives + negatives


def stratified(frame: pd.DataFrame, key: str, budget: int, label: pd.Series | None) -> list:
    """Seeded subset of unique keys, keeping the label balance of the eligible set."""
    keys = frame[key].drop_duplicates()
    if label is None or len(keys) <= budget:
        return sorted(keys, key=stable_hash)[:budget]
    per_key = frame.assign(_l=label).groupby(key)["_l"].max()
    chosen = []
    for value, group in per_key.groupby(per_key):
        quota = round(budget * len(group) / len(per_key))
        chosen += sorted(group.index, key=stable_hash)[:quota]
    return chosen[:budget]


class Genes:
    def __init__(self, path: Path):
        g = pd.read_feather(path, columns=["Chromosome", "Feature", "Start", "End", "Strand", "gene_id", "gene_type"])
        g["gene"] = g.gene_id.str.split(".").str[0]
        self.genes = g[g.Feature == "gene"].set_index("gene")
        self.exons = g[g.Feature == "exon"].groupby("gene")

    def exon_bins(self, gene: str, target_start: int) -> list[int] | None:
        if gene not in self.genes.index or gene not in self.exons.groups:
            return None
        bins = set()
        for start, end in self.exons.get_group(gene)[["Start", "End"]].itertuples(index=False):
            lo = max(0, (start - target_start) // BIN)
            hi = min(TARGET_BP // BIN, (end - target_start - 1) // BIN + 1)
            bins.update(range(lo, hi))
        return sorted(bins) or None

    def strand(self, gene: str) -> str:
        row = self.genes.loc[gene]
        return str((row.iloc[0] if isinstance(row, pd.DataFrame) else row).Strand)

    def tss(self, gene: str) -> tuple[str, int] | None:
        if gene not in self.genes.index:
            return None
        row = self.genes.loc[gene]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return row.Chromosome, int(row.Start if row.Strand == "+" else row.End - 1)


def window_start(center0: int) -> int:
    return center0 - INPUT_BP // 2


def slices(kind: str, pos0: int, input_start: int) -> dict:
    """Output slices the model must return, in target-window coordinates (dense bins / splice bp)."""
    offset = pos0 - (input_start + TARGET_OFFSET)  # edited base, relative to the target window
    if kind in ("eqtl", "enhancer"):
        return dict(dense_bin_start=0, dense_bin_end=TARGET_BP // BIN, splice_start=0, splice_end=0)
    if kind == "splice":
        return dict(dense_bin_start=0, dense_bin_end=0,
                    splice_start=max(0, offset - SPLICE_HALF_WINDOW),
                    splice_end=min(TARGET_BP, offset + SPLICE_HALF_WINDOW + 1))
    half = CENTER_HALF_WIDTH if kind == "center" else TRAITGYM_HALF_WINDOW
    return dict(dense_bin_start=max(0, (offset - half) // BIN),
                dense_bin_end=min(TARGET_BP // BIN, (offset + half) // BIN + 1),
                splice_start=0, splice_end=0)


def build_split(split: str, ctx: dict, test_keys: dict | None) -> tuple[list, dict, dict, dict]:
    """Queries and labels of one split. `test_keys` (valid only) holds the keys the test split used."""
    cfg = SPLITS[split]
    fasta, genes, sources = ctx["fasta"], ctx["genes"], ctx["sources"]
    regions, test_fold = ctx["regions"][cfg["fold"]], ctx["regions"]["fold1"]
    perturbations, labels, summary, chosen = [], {}, {}, {}

    def add_snv(eval_name, chrom, pos0, allele_a, allele_b, families, kind, center0=None):
        center0 = pos0 if center0 is None else center0
        start = window_start(center0)
        if start < 0 or start + INPUT_BP > fasta.get_reference_length(chrom):
            return None
        pid = f"{eval_name}:{chrom}:{pos0}:{allele_a}>{allele_b}:{center0}"
        perturbations.append(dict(perturbation_id=pid, eval=eval_name, chrom=chrom, input_start=start,
                                  edit="snv", edit_start=pos0, edit_end=pos0 + 1, allele_a=allele_a,
                                  allele_b=allele_b, families=families, **slices(kind, pos0, start)))
        return pid

    def restrict(name, table, key, chrom, pos, label):
        """Rows of the split's pool. Test: fold1 (or genome-wide). Valid: fold0, else non-test loci."""
        if split == "test":
            if name in GENOME_WIDE:
                return table, "genome-wide"
            return table[[in_fold(regions, c, int(x)) for c, x in zip(chrom, pos)]], "fold1"
        table = table[~table[key].isin(test_keys.get(name, ()))]
        chrom, pos = chrom.loc[table.index], pos.loc[table.index]
        label = None if label is None else label.loc[table.index]
        own = [in_fold(regions, c, int(x)) for c, x in zip(chrom, pos)]
        if name not in GENOME_WIDE and (label is None or int(label[own].sum()) >= VALID_POSITIVES):
            return table[own], "fold0"
        return table[[not in_fold(test_fold, c, int(x)) for c, x in zip(chrom, pos)]], "outside fold1"

    for name, (source, kind, budget, family) in EVALS.items():
        budget = cfg["budget"] or budget
        table = pd.read_feather(sources / "ag-evals" / f"{source}.feather")
        table = table.rename(columns={"prediction": "alphagenome_prediction"})
        if kind == "enhancer":
            table = table[table.was_scored.astype(bool)].copy()
            parts = table.enhancer_id.str.extract(r"_(chr[^_]+)_(\d+):(\d+)$")
            table["chrom"], table["e_start"], table["e_end"] = parts[0], parts[1].astype(int), parts[2].astype(int)
            table["gene"] = table.gene_id.str.split(".").str[0]
            tss = table.gene.map(lambda g: genes.tss(g))
            table = table[tss.notna()].copy()
            table["tss"] = [t[1] for t in tss[tss.notna()]]
            mid = (table.e_start + table.e_end) // 2
            table = table[(mid - table.tss).abs() < INPUT_BP // 2 - 20_000]
            table["pair"] = table.enhancer_id + "|" + table.gene
            table, pool = restrict(name, table, "pair", table.chrom, (table.e_start + table.e_end) // 2,
                                   table.target > 0)
            keep = fixed_design(table, "pair", table.target > 0, cfg["positives"])
            chosen[name] = set(keep)
            table = table[table.pair.isin(keep)].copy()
            rows = []
            for r in table.itertuples(index=False):
                start = window_start(r.tss)
                bins = genes.exon_bins(r.gene, start + TARGET_OFFSET)
                if bins is None:
                    continue
                pid = f"{name}:{r.chrom}:{r.e_start}-{r.e_end}:{r.tss}"
                perturbations.append(dict(perturbation_id=pid, eval=name, chrom=r.chrom, input_start=start,
                                          edit="mask", edit_start=int(r.e_start), edit_end=int(r.e_end),
                                          allele_a="", allele_b="N", families=[family],
                                          **slices("enhancer", r.tss, start)))
                rows.append(dict(perturbation_id=pid, gene=r.gene, gene_strand=genes.strand(r.gene),
                                 exon_bins=bins, target=float(r.target),
                                 tss_distance=float(r.tss_distance), alphagenome_prediction=r.alphagenome_prediction))
            labels[name] = pd.DataFrame(rows)
            summary[name] = dict(eligible=len(table), rows=len(rows), perturbations=len(rows), pool=pool)
            continue

        parsed = table.variant_id.map(parse_variant)
        table["chrom"] = [p[0] for p in parsed]; table["pos0"] = [p[1] for p in parsed]
        table["allele_a"] = [p[2] for p in parsed]; table["allele_b"] = [p[3] for p in parsed]
        table = table[(table.allele_a.str.len() == 1) & (table.allele_b.str.len() == 1)]
        if name == "mfass":
            table = table[table.subset == "test"]
        label = table.target.fillna(-1) > 0 if name != "eqtl_coefficient" else None
        table, pool = restrict(name, table, "variant_id", table.chrom, table.pos0, label)
        label = None if label is None else label.loc[table.index]
        keep = (fixed_design(table, "variant_id", label, cfg["positives"]) if name in FIXED_DESIGN
                else stratified(table, "variant_id", budget, label))
        chosen[name] = set(keep)
        table = table[table.variant_id.isin(keep)].copy()
        rows = []
        for vid, group in table.groupby("variant_id", sort=True):
            r = group.iloc[0]
            pid = add_snv(name, r.chrom, int(r.pos0), r.allele_a, r.allele_b, [family], kind)
            if pid is None:
                continue
            target_start = window_start(int(r.pos0)) + TARGET_OFFSET
            for g in group.itertuples(index=False):
                row = dict(perturbation_id=pid, variant_id=vid, target=g.target,
                           alphagenome_prediction=g.alphagenome_prediction)
                if kind == "eqtl":
                    gene = str(g.gene_id).split(".")[0]
                    bins = genes.exon_bins(gene, target_start)
                    if bins is None:
                        continue
                    row.update(gene=gene, gene_strand=genes.strand(gene), tissue=g.tissue, exon_bins=bins)
                elif "tissue" in group:
                    row.update(tissue=g.tissue)
                rows.append(row)
        labels[name] = pd.DataFrame(rows)
        summary[name] = dict(eligible_variants=int(len(set(keep))), rows=len(rows),
                             perturbations=int(labels[name].perturbation_id.nunique()) if rows else 0,
                             pool=pool)

    # TraitGym: Mendelian and complex-trait causal regulatory variants vs matched controls.
    for name, source in (("traitgym_mendelian", "mendelian_traits_matched_9"),
                         ("traitgym_complex", "complex_traits_matched_9")):
        table = pd.read_parquet(sources / "traitgym" / f"{source}.test.parquet")
        anchors = {m: pd.read_parquet(sources / "traitgym" / f"{source}.{m}.parquet").iloc[:, 0].to_numpy()
                   for m in ("Borzoi_L2_L2.plus.all", "Enformer_L2_L2.plus.all", "CADD.plus.RawScore",
                             "GPN-MSA_absLLR.plus.score")}
        for model, values in anchors.items():
            table[model] = values
        table["chrom"] = "chr" + table.chrom.astype(str); table["pos0"] = table.pos - 1
        # Not restricted to a fold: the label (causal for a trait) appears nowhere in training, and
        # fold1 holds only 35 Mendelian / 17 complex causal variants. AlphaGenome likewise scores
        # variants with models trained on every fold. Whole match groups are kept, so each causal
        # variant stays with its nine matched controls; valid takes groups the test split did not.
        groups = sorted(table.match_group.unique(), key=stable_hash)
        if split == "valid":
            groups = [g for g in groups if g not in test_keys[name]]
        chosen[name] = set(groups[:cfg["match_groups"]])
        table = table[table.match_group.isin(chosen[name])]
        rows = []
        dense = ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]
        for r in table.to_dict("records"):
            pid = add_snv(name, r["chrom"], int(r["pos0"]), r["ref"], r["alt"], dense, "traitgym")
            if pid:
                rows.append(dict(perturbation_id=pid, chrom=r["chrom"], label=bool(r["label"]),
                                 match_group=r["match_group"], **{f"anchor_{m}": float(r[m]) for m in anchors}))
        labels[name] = pd.DataFrame(rows)
        summary[name] = dict(rows=len(rows), positives=int(sum(x["label"] for x in rows)), perturbations=len(rows),
                             pool="genome-wide" if split == "test" else "genome-wide, test groups excluded")
    return perturbations, labels, summary, chosen


def write_split(out: Path, perturbations: list, labels: dict, tracks: pd.DataFrame) -> int:
    (out / "public").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(perturbations).drop_duplicates("perturbation_id")
    frame.to_parquet(out / "public/perturbations.parquet", index=False)
    for name, table in labels.items():
        table.to_parquet(out / "labels" / f"{name}.parquet", index=False)
    tracks.to_parquet(out / "labels/tracks.parquet", index=False)
    return len(frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True, help="materialized release (metadata/)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import pysam
    bed = a.release / "metadata/sequences_human.bed.gz"
    ctx = dict(fasta=pysam.FastaFile(str(a.release / "reference/GRCh38.p13.genome.fa")),
               genes=Genes(a.sources / "gencode.v46.annotation.gtf.gz.feather"), sources=a.sources,
               regions={fold: fold_regions(bed, fold) for fold in ("fold0", "fold1")})
    tracks = pd.read_parquet(a.release / "metadata/tracks.parquet")
    # Test (verifier only): <output>/public + <output>/labels. Valid (agent-visible): <output>/valid.
    test = build_split("test", ctx, None)
    valid = build_split("valid", ctx, test[3])
    overlap = {name: len(test[3][name] & valid[3][name]) for name in test[3]}
    assert not any(overlap.values()), overlap
    summary = dict(seed=SEED, perturbations=write_split(a.output, test[0], test[1], tracks), evals=test[2],
                   valid=dict(perturbations=write_split(a.output / "valid", valid[0], valid[1], tracks),
                              evals=valid[2]),
                   windows=dict(splice_half_window_bp=SPLICE_HALF_WINDOW, center_half_width_bp=CENTER_HALF_WIDTH,
                                traitgym_half_window_bp=TRAITGYM_HALF_WINDOW))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
