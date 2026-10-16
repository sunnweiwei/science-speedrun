#!/usr/bin/env python3
"""Split the FoldBench suite into a validation part (visible to the agent) and a test part (verifier only).

    split_suite.py SOURCE_FOLDBENCH_DIR OUTPUT_ROOT [--max-per-track 10] [--validation-fraction 0.2]
        [--seed foldbench-split-v1]

SOURCE_FOLDBENCH_DIR is the materialized `foldbench/` directory (suite/, ground_truths/,
features.esmc6b_final_int8.jsonl, features/). The script writes OUTPUT_ROOT/foldbench-val and
OUTPUT_ROOT/foldbench-test, each a complete suite directory with the same layout, so every tool
that reads `foldbench/` reads either part unchanged. The source is not modified.

Assemblies are grouped before they are assigned, so no near-duplicate straddles the split: two
assemblies share a group when they come from the same PDB entry, or when any of their polymer
chains are identical or similar (Jaccard similarity of k-mer sets at least 0.3; k = 5 for protein,
6 for RNA/DNA; k-mers shared by more than 50 sequences are ignored as low-complexity). Groups are
taken in a seeded hash order, and a group joins validation only while every track it touches is
below its quota, min(max_per_track, ceil(fraction x assemblies in that track)); the rest is test.
The validation part is a small set of examples per track, for seeing what each track asks and
checking a package end to end; the test part is what the verifier scores. The split is a pure
function of the suite and the seed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

POLYMERS = {"protein": 5, "rna": 6, "dna": 6}
SIMILARITY = 0.3
COMMON_KMER = 50


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_requests(suite: Path) -> tuple[dict, dict]:
    manifest = json.loads((suite / "suite_manifest.json").read_text())
    requests = {item["request_id"]: json.loads((suite / item["path"]).read_text()) for item in manifest["requests"]}
    return manifest, requests


def groups_of(manifest: dict, requests: dict) -> list[list[str]]:
    """Union-find over assemblies: same PDB entry, or a shared identical or similar polymer chain."""
    ids = [item["request_id"] for item in manifest["requests"]]
    parent = {rid: rid for rid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_entry = collections.defaultdict(list)
    for rid in ids:
        by_entry[rid.split("-")[0].lower()].append(rid)
    for members in by_entry.values():
        for other in members[1:]:
            union(members[0], other)

    owners = collections.defaultdict(set)            # (type, sequence) -> request ids
    for rid in ids:
        for entity in requests[rid]["target"]["entities"]:
            if entity["type"] in POLYMERS and entity.get("sequence"):
                owners[(entity["type"], entity["sequence"].upper())].add(rid)
    sequences = list(owners)
    for key in sequences:                            # identical chains
        members = sorted(owners[key])
        for other in members[1:]:
            union(members[0], other)

    kmers = {}
    index = collections.defaultdict(set)
    for n, (kind, seq) in enumerate(sequences):
        k = POLYMERS[kind]
        kmers[n] = {seq[i:i + k] for i in range(len(seq) - k + 1)}
        for kmer in kmers[n]:
            index[(kind, kmer)].add(n)
    for n, (kind, _) in enumerate(sequences):        # similar chains
        shared = collections.Counter()
        for kmer in kmers[n]:
            holders = index[(kind, kmer)]
            if len(holders) <= COMMON_KMER:
                for m in holders:
                    if m > n:
                        shared[m] += 1
        for m, count in shared.items():
            union_size = len(kmers[n]) + len(kmers[m]) - count
            if union_size and count / union_size >= SIMILARITY:
                union(min(owners[sequences[n]]), min(owners[sequences[m]]))

    grouped = collections.defaultdict(list)
    for rid in ids:
        grouped[find(rid)].append(rid)
    return [sorted(members) for members in grouped.values()]


def assign(manifest: dict, groups: list[list[str]], fraction: float, seed: str, max_per_track: int) -> set[str]:
    tracks = {item["request_id"]: item["tracks"] for item in manifest["requests"]}
    quota = {track: min(max_per_track, math.ceil(fraction * count))
             for track, count in manifest["assemblies_by_track"].items()}
    used = collections.Counter()
    validation = set()
    for members in sorted(groups, key=lambda g: sha256_text(f"{seed}:{g[0]}")):
        need = collections.Counter(track for rid in members for track in tracks[rid])
        if all(used[track] + count <= quota[track] for track, count in need.items()):
            validation.update(members)
            used.update(need)
    return validation


def write_part(source: Path, dest: Path, manifest: dict, requests: dict, keep: set[str], name: str,
               fraction: float, seed: str, source_sha: str, max_per_track: int) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    suite_out = dest / "suite"
    (suite_out / "requests").mkdir(parents=True)
    (suite_out / "targets").mkdir()
    (dest / "ground_truths").mkdir()
    items = [item for item in manifest["requests"] if item["request_id"] in keep]
    for item in items:
        shutil.copy2(source / "suite" / item["path"], suite_out / item["path"])
        shutil.copy2(source / "ground_truths" / f"{item['request_id']}.cif", dest / "ground_truths")
    targets_by_track = {}
    for table in sorted((source / "suite" / "targets").glob("*.csv")):
        with table.open(newline="") as stream:
            rows = list(csv.reader(stream))
        kept = [row for row in rows[1:] if row and row[0] in keep]
        with (suite_out / "targets" / table.name).open("w", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows([rows[0]] + kept)
        targets_by_track[table.stem] = len(kept)
    proteins = {sha256_text(entity["sequence"]) for rid in keep for entity in requests[rid]["target"]["entities"]
                if entity["type"] == "protein"}
    feature_rows = []
    for line in (source / "features.esmc6b_final_int8.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["sequence_sha256"] in proteins:
            feature_rows.append(line)
            target = dest / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row["path"], target)
    (dest / "features.esmc6b_final_int8.jsonl").write_text("".join(f"{line}\n" for line in feature_rows))
    by_track = collections.Counter(track for item in items for track in item["tracks"])
    out = dict(manifest, requests=items, assemblies=len(items),
               assemblies_by_track={track: by_track.get(track, 0) for track in manifest["assemblies_by_track"]},
               targets_by_track=targets_by_track,
               split={"name": name, "validation_fraction": fraction, "max_per_track": max_per_track, "seed": seed,
                      "source_suite_manifest_sha256": source_sha,
                      "grouping": f"same PDB entry, or identical/similar polymer chains (k-mer Jaccard >= {SIMILARITY})"})
    (suite_out / "suite_manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return {"assemblies": len(items), "assemblies_by_track": out["assemblies_by_track"],
            "targets_by_track": targets_by_track, "protein_features": len(feature_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-per-track", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="foldbench-split-v1")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest, requests = load_requests(source / "suite")
    source_sha = hashlib.sha256((source / "suite" / "suite_manifest.json").read_bytes()).hexdigest()
    groups = groups_of(manifest, requests)
    validation = assign(manifest, groups, args.validation_fraction, args.seed, args.max_per_track)
    test = {item["request_id"] for item in manifest["requests"]} - validation
    report = {"groups": len(groups), "largest_group": max(len(g) for g in groups)}
    for name, keep in (("validation", validation), ("test", test)):
        dest = args.output_root / ("foldbench-val" if name == "validation" else "foldbench-test")
        report[name] = write_part(source, dest, manifest, requests, keep, name,
                                  args.validation_fraction, args.seed, source_sha, args.max_per_track)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
