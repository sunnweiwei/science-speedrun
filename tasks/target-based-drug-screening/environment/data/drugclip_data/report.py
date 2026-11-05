from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_protocol


def _distribution(series: pd.Series) -> str:
    if series.empty:
        return "n/a"
    values = series.dropna().astype(float)
    return ", ".join(f"{name}={value:.3g}" for name, value in
                     [("min", values.min()), ("p25", values.quantile(.25)),
                      ("median", values.median()), ("p75", values.quantile(.75)),
                      ("max", values.max()), ("mean", values.mean())])


def build_report(data_root: Path, smoke: bool = False) -> None:
    protocol = load_protocol()
    profsa = pd.read_parquet(data_root / "canonical" / "profsadb.parquet")
    biolip = pd.read_parquet(data_root / "canonical" / "biolip2.parquet")
    targets = pd.read_parquet(data_root / "test" / "test_targets.parquet")
    ligands = pd.read_parquet(data_root / "test" / "test_ligands.parquet",
                              columns=["benchmark", "target_id", "label"])
    clean_bio = biolip[biolip.exclusion_reason.eq("")]
    valid_conformer_count = int(biolip.exclusion_reason.isin([
        "", "test_protein_similarity", "exact_test_active_match",
        "test_active_ecfp4_similarity",
    ]).sum())
    valid_pocket_count = valid_conformer_count + int(
        biolip.exclusion_reason.eq("invalid_ligand_conformer").sum())
    valid_complex_count = valid_pocket_count + int(
        biolip.exclusion_reason.eq("invalid_or_empty_pocket").sum())
    exclusions = biolip.loc[biolip.exclusion_reason.ne(""), "exclusion_reason"].value_counts()
    profsa_exclusions = profsa.loc[profsa.exclusion_reason.ne(""), "exclusion_reason"].value_counts()
    lines = [
        "# Target-based drug-screening dataset report", "",
        f"Build mode: **{'SMOKE / NOT FOR TRAINING' if smoke else 'FULL'}**.", "",
        "## Fixed protocol", "",
        f"- seed: `{protocol['seed']}`",
        f"- pocket: any protein residue with any atom ≤ `{protocol['pocket_cutoff_angstrom']} Å` from any observed ligand atom",
        f"- protein leakage: MMseqs2 pident strictly > `{protocol['protein_identity_threshold_percent']}%`, with `-c {protocol['protein_coverage_threshold']} --cov-mode {protocol['mmseqs_cov_mode']}` (both query and target coverage)",
        f"- ligand leakage: Morgan/ECFP4 radius `{protocol['morgan_radius']}`, `{protocol['morgan_nbits']}` bits, Tanimoto strictly > `{protocol['ligand_tanimoto_threshold']}` or exact InChIKey",
        f"- ligand conformers: up to `{protocol['max_ligand_conformers']}`, deterministic ETKDGv3, MMFF94s (UFF fallback), prune RMS `{protocol['conformer_prune_rms_angstrom']} Å`",
        "",
        "## Frozen tests", "",
    ]
    for benchmark in ("dude", "lit_pcba"):
        selected_targets = targets[targets.benchmark.eq(benchmark)]
        selected_ligands = ligands[ligands.benchmark.eq(benchmark)]
        lines.append(f"- {benchmark}: {selected_targets.target_name.nunique()} targets, "
                     f"{len(selected_targets)} receptor sequences, "
                     f"{int(selected_ligands.label.eq(1).sum())} actives, "
                     f"{int(selected_ligands.label.eq(0).sum())} decoys/inactives")
    lines += ["", "Archive URLs, byte sizes, SHA256 values, and pinned Hugging Face revision are in `test/manifest.json`.", "",
              "## ProFSADB flow", "",
              f"- raw parsed samples: {len(profsa)}",
              f"- removed due to test-protein contamination: {int(profsa.exclusion_reason.eq('test_protein_similarity').sum())}",
              f"- other invalid/unresolved exclusions: {int((profsa.exclusion_reason.ne('') & profsa.exclusion_reason.ne('test_protein_similarity')).sum())}",
              f"- remaining clean pool: {int(profsa.exclusion_reason.eq('').sum())}",
              f"- selected fixed subset: {int(profsa.selected.sum())}", ""]
    if len(profsa_exclusions):
        lines += ["Exclusions: " + ", ".join(f"`{key}`={value}" for key, value in profsa_exclusions.items()), ""]
    lines += ["## BioLiP2/Q-BioLiP flow", "",
              f"- raw annotation records: {len(biolip)}",
              f"- valid protein-small-molecule complexes: {valid_complex_count}",
              f"- valid 6 Å pockets: {valid_pocket_count}",
              f"- valid ligand conformers: {valid_conformer_count}",
              f"- removed due to test-protein similarity: {int(biolip.exclusion_reason.eq('test_protein_similarity').sum())}",
              f"- removed due to exact/similar test ligand: {int(biolip.exclusion_reason.isin(['exact_test_active_match','test_active_ecfp4_similarity']).sum())}",
              f"- final clean pool: {len(clean_bio)}",
              f"- train: {int(biolip.split.eq('train').sum())}",
              f"- dev: {int(biolip.split.eq('dev').sum())}", ""]
    if len(exclusions):
        lines += ["Exclusions: " + ", ".join(f"`{key}`={value}" for key, value in exclusions.items()), ""]
    lines += ["## Clean BioLiP2 distributions", "",
              f"- pocket atom count: {_distribution(clean_bio.pocket_atom_count)}",
              f"- pocket residue count: {_distribution(clean_bio.pocket_residue_count)}",
              f"- ligand heavy atom count: {_distribution(clean_bio.ligand_heavy_atom_count)}",
              f"- protein length: {_distribution(clean_bio.protein_sequence.str.len())}",
              f"- ligand conformers: {_distribution(clean_bio.ligand_conformer_count)}",
              f"- max test-protein identity (%): {_distribution(clean_bio.max_test_sequence_identity)}",
              f"- max test-active Tanimoto: {_distribution(clean_bio.max_test_ligand_tanimoto)}", "",
              "## Regeneration", "",
              "```bash", "cd target-based-drug-screening/environment/data",
              "# Default: /home/zifengw2/github/rsi-for-science/.agent-runtime/target-based-drug-screening/materialized",
              "bash materialize.sh", "```", "",
              "For an offline structural dry run:", "", "```bash",
              "bash materialize.sh /tmp/target-screening-smoke --smoke --overwrite", "```", ""]
    (data_root / "DATASET_REPORT.md").write_text("\n".join(lines))
