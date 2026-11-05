from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from .io import require_executable, run
from .structure import write_fasta

FORMAT = "query,target,pident,qcov,tcov,alnlen,qlen,tlen,evalue,bits"


def search(queries: list[tuple[str, str]], targets_fasta: Path, output: Path,
           identity_percent: float, coverage: float, cov_mode: int) -> dict[str, dict]:
    """Return the best qualifying MMseqs2 hit for every query.

    MMseqs reports pident as percent and qcov/tcov as fractions in the requested TSV. The
    command applies `-c coverage --cov-mode 0`; mode 0 requires alignment
    coverage on both query and target. Contamination is strict pident > the
    configured threshold (an exact 30.0% hit is retained).
    """
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
    best: dict[str, dict] = {}
    if not output.exists():
        return best
    with output.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 10:
                continue
            hit = {
                "query": row[0], "target": row[1], "pident": float(row[2]),
                "qcov": float(row[3]), "tcov": float(row[4]),
                "alnlen": int(row[5]), "qlen": int(row[6]), "tlen": int(row[7]),
                "evalue": float(row[8]), "bits": float(row[9]),
            }
            previous = best.get(hit["query"])
            if previous is None or (hit["pident"], min(hit["qcov"], hit["tcov"]), hit["bits"]) > (
                previous["pident"], min(previous["qcov"], previous["tcov"]), previous["bits"]
            ):
                best[hit["query"]] = hit
    return best


def cluster(sequences: list[tuple[str, str]], output_prefix: Path,
            identity_percent: float, coverage: float, cov_mode: int) -> dict[str, str]:
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
