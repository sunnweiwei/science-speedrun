"""Audit the published S28 target counts using the v1 supplement as the source."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from pathlib import Path
import re

REFERENCE = Path(__file__).with_name("target_counts.csv")
PDF_SHA256 = "cf7b450d9a9b56898bd1730f986bfd6b945a56a61cc33f8f5a9ea0cce7c5440a"
HOURS_PER_TARGET = 4
FIELDS = ("target", "naive_passers", "naive_clusters", "search_passers", "search_clusters")
TARGETS = {"DFHBI", "PLP", "apixaban", "bilirubin", "chloramphenicol", "dopamine",
           "methotrexate", "SAM", "serotonin", "thyroxine"}


def rate(count, hours):
    return (Decimal(count) / Decimal(hours)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def read_reference():
    with REFERENCE.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected target-count columns")
        rows = {}
        for record in reader:
            name = record["target"]
            if name in rows:
                raise ValueError(f"Duplicate target: {name}")
            rows[name] = tuple(int(record[k]) for k in FIELDS[1:])
    return rows


def extract_table(path):
    import pymupdf  # Only needed to compare directly with the source PDF.

    if hashlib.sha256(path.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("Expected the pinned bioRxiv v1 supplementary PDF")
    with pymupdf.open(path) as document:
        pages = [page.get_text(sort=True) for page in document]
    page, = (p for p in pages if "Target-specific pass rates on a 4-GPU-hour" in p
             and "Pooled" in p and "thyroxine" in p)
    lines = page.split("Target-specific pass rates on a 4-GPU-hour", 1)[1].splitlines()
    rows = {}
    pooled = None
    for line in lines:
        match = re.fullmatch(r"\s*(\w+)\s+([\d,]+)\s+([\d,]+)\s+(\d+\.\d+)\s+"
                             r"([\d,]+)\s+([\d,]+)\s+(\d+\.\d+)\s*", line)
        if not match:
            continue
        name, p, c, r, q, d, s = match.groups()
        counts = tuple(int(v.replace(",", "")) for v in (p, c, q, d))
        hours = HOURS_PER_TARGET * (len(TARGETS) if name == "Pooled" else 1)
        if (rate(counts[1], hours), rate(counts[3], hours)) != (Decimal(r), Decimal(s)):
            raise ValueError(f"Inconsistent printed throughput: {name}")
        if name == "Pooled":
            pooled = counts
            break
        if name in rows:
            raise ValueError(f"Duplicate PDF target: {name}")
        rows[name] = counts
    if pooled != tuple(sum(column) for column in zip(*rows.values())):
        raise ValueError("Pooled PDF counts disagree with the target rows")
    return rows


def summarize(rows):
    if set(rows) != TARGETS:
        raise ValueError("Expected all ten S28 targets, including zero-yield targets")
    for name, (p, c, q, d) in rows.items():
        if not (0 <= c <= p and 0 <= d <= q):
            raise ValueError(f"Inconsistent pass and cluster counts: {name}")
    sums = tuple(sum(column) for column in zip(*rows.values()))
    p, c, q, d = sums
    hours = len(rows) * HOURS_PER_TARGET
    return {
        "gpu_hours_per_arm": hours,
        "naive": {"passers": p, "clusters": c, "clusters_per_gpu_hour": str(rate(c, hours))},
        "interaction_search": {"passers": q, "clusters": d, "clusters_per_gpu_hour": str(rate(d, hours))},
        "relative_gain_percent": str((Decimal(100) * (d - c) / c).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
        "target_wins_ties_losses": [sum(r[3] > r[1] for r in rows.values()),
                                   sum(r[3] == r[1] for r in rows.values()),
                                   sum(r[3] < r[1] for r in rows.values())],
    }


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, help="Check every bundled count and printed rate against the v1 PDF")
    args = parser.parse_args()
    reference = read_reference()
    if args.supplement and extract_table(args.supplement) != reference:
        raise SystemExit("Bundled S28 counts differ from the paper")
    print(json.dumps(summarize(reference), indent=2))
