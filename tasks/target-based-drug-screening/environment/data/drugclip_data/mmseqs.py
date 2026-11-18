from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Iterable

FORMAT = "query,target,pident,qcov,tcov,alnlen,qlen,tlen,evalue,bits"


def alignment_coverage(hit: dict, cov_mode: int) -> float:
    """Return the effective alignment coverage fraction under MMseqs2 cov_mode.

    - cov_mode == 0: bidirectional min(qcov, tcov)
    - cov_mode == 1: unidirectional target coverage tcov (test pocket coverage)
    - cov_mode == 2: unidirectional query coverage qcov
    """
    if cov_mode == 1:
        return float(hit["tcov"])
    if cov_mode == 2:
        return float(hit["qcov"])
    if cov_mode == 0:
        return float(min(hit["qcov"], hit["tcov"]))
    raise ValueError(f"unsupported mmseqs cov_mode: {cov_mode}")


def is_contaminated_hit(
    hit: dict | None,
    identity_percent: float,
    coverage: float,
    cov_mode: int,
) -> bool:
    """Return True iff `hit` satisfies alignment coverage >= `coverage` under
    `cov_mode` and strict sequence identity `pident > identity_percent`."""
    if not hit:
        return False
    pident = float(hit["pident"]) if "pident" in hit else float(hit["fident"]) * 100.0
    return alignment_coverage(hit, cov_mode) >= coverage and pident > identity_percent


def select_best_hits(
    rows: Iterable[dict],
    coverage: float,
    cov_mode: int,
) -> dict[str, dict]:
    """Select the highest-identity qualifying hit per query under `cov_mode`."""
    best: dict[str, dict] = {}
    for raw in rows:
        pident = float(raw["pident"]) if "pident" in raw else float(raw["fident"]) * 100.0
        hit = {
            **raw,
            "query": str(raw["query"]),
            "target": str(raw["target"]),
            "pident": pident,
            "qcov": float(raw["qcov"]),
            "tcov": float(raw["tcov"]),
            "alnlen": int(raw.get("alnlen", 0)),
            "qlen": int(raw["qlen"]),
            "tlen": int(raw["tlen"]),
            "evalue": float(raw.get("evalue", 0.0)),
            "bits": float(raw.get("bits", 0.0)),
        }
        cov = alignment_coverage(hit, cov_mode)
        if cov < coverage:
            continue
        previous = best.get(hit["query"])
        if previous is None or (hit["pident"], cov, hit["bits"]) > (
            previous["pident"], alignment_coverage(previous, cov_mode), previous["bits"]
        ):
            best[hit["query"]] = hit
    return best


def parse_mmseqs_tsv(output: Path, coverage: float, cov_mode: int) -> dict[str, dict]:
    if not output.exists():
        return {}
    parsed = []
    with output.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 10:
                continue
            parsed.append({
                "query": row[0], "target": row[1], "pident": float(row[2]),
                "qcov": float(row[3]), "tcov": float(row[4]),
                "alnlen": int(row[5]), "qlen": int(row[6]), "tlen": int(row[7]),
                "evalue": float(row[8]), "bits": float(row[9]),
            })
    return select_best_hits(parsed, coverage=coverage, cov_mode=cov_mode)


def search(queries: list[tuple[str, str]], targets_fasta: Path, output: Path,
           identity_percent: float, coverage: float, cov_mode: int) -> dict[str, dict]:
    """Return the best qualifying MMseqs2 hit for every query.

    MMseqs reports pident as percent and qcov/tcov as fractions in the requested TSV.
    With `cov_mode=1` (`-c coverage --cov-mode 1`), alignments must cover at least
    `coverage` of the target test pocket domain (`tcov >= coverage`), preventing
    multi-domain training proteins from leaking homologous test pocket domains.
    Contamination is strict `pident > identity_percent` (an exact 30.0% hit is retained).
    """
    from .io import require_executable, run
    from .structure import write_fasta

    mmseqs = require_executable("mmseqs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="drugclip-mmseqs-") as tmp:
        tmp_path = Path(tmp)
        query_fasta = tmp_path / "query.fasta"
        write_fasta(query_fasta, queries)
        run([
            mmseqs, "easy-search", str(query_fasta), str(targets_fasta), str(output),
            str(tmp_path / "work"), "--min-seq-id", "0.0",
            "-c", str(coverage), "--cov-mode", str(cov_mode),
            "--format-output", FORMAT, "--threads", "1",
        ])
    return parse_mmseqs_tsv(output, coverage=coverage, cov_mode=cov_mode)


def cluster(sequences: list[tuple[str, str]], output_prefix: Path,
            identity_percent: float, coverage: float, cov_mode: int) -> dict[str, str]:
    from .io import require_executable, run
    from .structure import write_fasta

    mmseqs = require_executable("mmseqs")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="drugclip-cluster-") as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "proteins.fasta"
        write_fasta(fasta, sequences)
        run([
            mmseqs, "easy-cluster", str(fasta), str(output_prefix), str(tmp_path / "work"),
            "--min-seq-id", str(identity_percent / 100.0), "-c", str(coverage),
            "--cov-mode", str(cov_mode), "--cluster-mode", "2", "--threads", "1",
        ])
    clusters: dict[str, str] = {}
    tsv = Path(str(output_prefix) + "_cluster.tsv")
    with tsv.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            clusters[member] = representative
    missing = {name for name, _ in sequences} - set(clusters)
    if missing:
        raise RuntimeError(f"MMseqs omitted {len(missing)} protein(s)")
    return clusters
