"""Read outcome columns from the v1 supplement. Each study retains its own units."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
from pathlib import Path
import re

from benchmark_curve import HERE, operating_points

SHA256 = "cf7b450d9a9b56898bd1730f986bfd6b945a56a61cc33f8f5a9ea0cce7c5440a"
COLUMNS = ("table", "comparison", "method", "steps", "clustered_pass_pct", "reported_seconds",
           "mean_hbond_count", "completed", "passing_designs", "passing_clusters", "gpu_hours", "clusters_h")


def read_paper(path):
    import pymupdf

    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Unexpected v1 supplementary PDF digest")
    with pymupdf.open(path) as doc:
        pages = [page.get_text(sort=True) for page in doc]

    def page_for(caption):
        page, = (text for text in pages if caption in text)
        return page

    records = []
    for table, caption, comparison, ncols in (
        ("S15", "Table S15 Unconditional sampling ablation", "Unconditional sampling", 2),
        ("S16", "Table S16 Hydrogen-bond search ablation", "Contacts in a separate search study", 1)
    ):
        section = page_for(caption).split(caption, 1)[0]
        for line in section.splitlines():
            m = re.match(r"^(Stochastic TSR(?: \([^)]+\))?|ODE TSR|Full-length SDE|Shortened SDE)\s{2,}", line)
            if m is None:
                continue
            cells = line[m.end():].split()
            metrics = (dict(clustered_pass_pct=cells[-2], reported_seconds=cells[-1]) if ncols == 2
                       else dict(mean_hbond_count=cells[-1]))
            records.append(dict(table=table, comparison=comparison, method=m[1], steps=int(cells[-ncols-2]), **metrics))

    def validate_yield(clusters, hours, printed):
        value = (Decimal(clusters) / Decimal(hours)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if value != Decimal(printed):
            raise ValueError(f"Printed rate {printed} disagrees with {clusters}/{hours}")

    for line in page_for("Table S18 Search comparison").splitlines():
        match = re.fullmatch(r"(Naive|Constant|Temporal)\s+([\d,]+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", line.strip())
        if match:
            method, completed, passes, clusters, hours, printed = match.groups()
            validate_yield(clusters, hours, printed)
            records.append(dict(table="S18", comparison="Single target", method=method,
                                completed=int(completed.replace(",", "")), passing_designs=passes,
                                passing_clusters=clusters, gpu_hours=hours, clusters_h=printed))
    for line in page_for("Table S22 Passing structural clusters").splitlines():
        match = re.fullmatch(r"(Regular|Burial > 85%)\s+(Naive best-of-N|Temporal|Constant criteria)\s+"
                             r"(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", line.strip())
        if match:
            cohort, method, clusters, hours, printed = match.groups()
            validate_yield(clusters, hours, printed)
            records.append(dict(table="S22", comparison=cohort, method=method,
                                passing_clusters=clusters, gpu_hours=hours, clusters_h=printed))
    expected_counts = {"S15": 5, "S16": 4, "S18": 3, "S22": 6}
    for table, count in expected_counts.items():
        keys = [(r["comparison"], r["method"]) for r in records if r["table"] == table]
        if len(set(keys)) != count or len(keys) != count:
            raise ValueError(f"Incomplete {table} extraction")

    points = {(p["track"], p["method"]): p for p in operating_points(HERE / "targets.csv")}
    seen = set()
    track = None
    for line in page_for("Table S29 Structure-predictor-guided search").splitlines():
        match = re.fullmatch(r"\s*(?:(Protein binding|Small-molecule binding|AME scaffolding|DNA binding)\s+)?"
                             r"(High thinking|Search)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*", line)
        if match:
            title, arm, *values = match.groups()
            if title:
                track = "Enzyme scaffolding" if title == "AME scaffolding" else title
            key = (track, "High" if arm == "High thinking" else arm)
            names = ("clustered_pct", "generation_clusters_h", "total_clusters_h")
            derived = [f"{points[key][name]:.1f}" for name in names]
            if key in seen or derived != values:
                raise ValueError(f"S29 disagrees with the independent target export for {key}")
            seen.add(key)
    if len(seen) != 8:
        raise ValueError("Incomplete S29 high/search comparison")
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return out.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supplement", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Update the published ablation outcome CSV")
    args = parser.parse_args()
    result = read_paper(args.supplement)
    dest = HERE / "ablation_results.csv"
    if args.refresh:
        dest.write_text(result)
    elif dest.read_text() != result:
        raise SystemExit("Ablation CSV differs from the pinned supplement")
    print("Checked 18 specialist-study rows and all 8 High/Search S29 rows against the v1 PDF.")
