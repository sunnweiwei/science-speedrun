"""Pinned AlphaGenome hg38 reference acquisition."""

from __future__ import annotations

from pathlib import Path

from .constants import (
    FAI_MD5_BASE64,
    FAI_SHA256,
    FAI_SIZE,
    FAI_URL,
    FASTA_MD5_BASE64,
    FASTA_SIZE,
    FASTA_URL,
)
from .downloads import download_verified


def fetch_fai(data_dir: Path) -> Path:
    return download_verified(
        FAI_URL,
        data_dir / "reference" / "GRCh38.p13.genome.fa.fai",
        expected_size=FAI_SIZE,
        expected_sha256=FAI_SHA256,
        expected_md5_base64=FAI_MD5_BASE64,
    )


def fetch_reference(data_dir: Path) -> tuple[Path, Path]:
    fasta = download_verified(
        FASTA_URL,
        data_dir / "reference" / "GRCh38.p13.genome.fa",
        expected_size=FASTA_SIZE,
        expected_md5_base64=FASTA_MD5_BASE64,
    )
    return fasta, fetch_fai(data_dir)
