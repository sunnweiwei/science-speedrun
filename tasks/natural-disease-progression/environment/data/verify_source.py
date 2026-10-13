"""Verify every MIMIC-IV source table used by v1 against the release manifest."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = ("hosp/patients.csv.gz", "hosp/admissions.csv.gz",
            "hosp/diagnoses_icd.csv.gz", "hosp/d_icd_diagnoses.csv.gz",
            "hosp/labevents.csv.gz", "hosp/prescriptions.csv.gz",
            "hosp/pharmacy.csv.gz", "hosp/emar.csv.gz")
RELEASE_MANIFEST_SHA256 = "b1f75ad5aba53104bce181da819b17fa1ef47f9f53ba09bde2cb7f8ca10f2675"


def verify(root: Path) -> dict:
    manifest = root / "SHA256SUMS.txt"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != RELEASE_MANIFEST_SHA256:
        raise ValueError("MIMIC-IV v3.1 release manifest checksum mismatch")
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    result = {}
    for name in REQUIRED:
        if name not in expected:
            raise ValueError(f"Missing release checksum: {name}")
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected[name]:
            raise ValueError(f"MIMIC-IV release checksum mismatch: {name}")
        result[name] = actual
    return {"source_version": "MIMIC-IV v3.1", "verified_files": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.mimic_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['verified_files'])} source tables against release SHA256SUMS.txt")
