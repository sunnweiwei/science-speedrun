#!/usr/bin/env python3
"""Materialize pinned PEER protein data into the peer-protein-input/v1 layout.

Downloads the public PEER/TAPE archives (pinned URL + md5), converts each LMDB
split to JSONL, and writes three views under the target directory:

  public/development/   fit/<task>.jsonl, tune/<task>.jsonl, stats.json, manifest.json
  public/verifier/      <task>.jsonl (inputs only), manifest.json
  private/              <task>.jsonl (labels only), manifest.json

Requires: python3, `pip install lmdb numpy`. Safe to rerun; downloads are
cached under <target>/cache and views are rebuilt atomically.
"""
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SCHEMA = "peer-protein-input/v1"
PRIVATE_SCHEMA = "peer-protein-private-labels/v1"
TAPE = "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch"
PEER = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com"

# One entry per task: pinned archive, inner LMDB directory name, source-split
# to view mapping (PEER-standard eval split), task kind, target field(s), and
# exact expected row counts.
#
# The counts were MEASURED from the pinned archives, not copied from torchdrug's
# dataset docstrings, which disagree with the data they distribute for 16 of the
# 33 splits. The three PPI docstrings report unique protein sequences rather
# than labeled pairs (human_ppi train: 6,844 unique proteins forming 35,669
# pairs), and stability/subcellular_localization/binary_localization appear to
# have been re-split upstream after the docstrings were written. All eleven md5
# checksums match, so these counts are what torchdrug itself loads today.
SPEC = {
    "fluorescence": dict(
        url=f"{TAPE}/fluorescence.tar.gz", md5="d63d1d51ec8c20ff0d981e4cbd67457a",
        inner="fluorescence", kind="regression", target="log_fluorescence",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 21446, "tune": 5362, "eval": 27217}),
    "stability": dict(
        url=f"{TAPE}/stability.tar.gz", md5="aa1e06eb5a59e0ecdae581e9ea029675",
        inner="stability", kind="regression", target="stability_score",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 53614, "tune": 2512, "eval": 12851}),
    "beta_lactamase": dict(
        url=f"{PEER}/peerdata/beta_lactamase.tar.gz", md5="65766a3969cc0e94b101d4063d204ba4",
        inner="beta_lactamase", kind="regression", target="scaled_effect1",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 4158, "tune": 520, "eval": 520}),
    "solubility": dict(
        url=f"{PEER}/peerdata/solubility.tar.gz", md5="8a8612b7bfa2ed80375db6e465ccf77e",
        inner="solubility", kind="classification", classes=2, target="solubility",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 62478, "tune": 6942, "eval": 1999}),
    "subcellular_localization": dict(
        url=f"{PEER}/peerdata/subcellular_localization.tar.gz", md5="37cb6138b8d4603512530458b7c8a77d",
        inner="subcellular_localization", kind="classification", classes=10, target="localization",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 8420, "tune": 2811, "eval": 2773}),
    "binary_localization": dict(
        url=f"{PEER}/peerdata/subcellular_localization_2.tar.gz", md5="5d2309bf1c0c2aed450102578e434f4e",
        inner="subcellular_localization_2", kind="classification", classes=2, target="localization",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 5184, "tune": 1729, "eval": 1749}),
    "fold": dict(
        url=f"{TAPE}/remote_homology.tar.gz", md5="1d687bdeb9e3866f77504d6079eed00a",
        inner="remote_homology", kind="classification", classes=1195, target="fold_label",
        splits={"train": "fit", "valid": "tune", "test_fold_holdout": "eval"},
        counts={"fit": 12312, "tune": 736, "eval": 718}),
    "secondary_structure": dict(
        url=f"{TAPE}/secondary_structure.tar.gz", md5="2f61e8e09c215c032ef5bc8b910c8e97",
        inner="secondary_structure", kind="ssp", target="ss3",
        splits={"train": "fit", "valid": "tune", "cb513": "eval"},
        counts={"fit": 8678, "tune": 2170, "eval": 513}),
    "yeast_ppi": dict(
        url=f"{PEER}/ppidata/yeast_ppi.zip", md5="3993b02c3080d74996cddf6fe798b1e8",
        inner="yeast_ppi", kind="pair_classification", classes=2, target="interaction",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 4945, "tune": 95, "eval": 394}),
    "human_ppi": dict(
        url=f"{PEER}/ppidata/human_ppi.zip", md5="89885545ebc2c11d774c342910230e20",
        inner="human_ppi", kind="pair_classification", classes=2, target="interaction",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 35669, "tune": 315, "eval": 237}),
    "ppi_affinity": dict(
        url=f"{PEER}/ppidata/ppi_affinity.zip", md5="d114907fd20c75820e41881f8901e9e4",
        inner="ppi_affinity", kind="pair_regression", target="interaction",
        splits={"train": "fit", "valid": "tune", "test": "eval"},
        counts={"fit": 2421, "tune": 203, "eval": 326}),
}
TASKS = tuple(SPEC)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def download(url, destination, expected_md5):
    if destination.is_file() and md5(destination) == expected_md5:
        return destination
    print(f"downloading {url}", flush=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    if md5(partial) != expected_md5:
        partial.unlink()
        raise ValueError(f"md5 mismatch for {url}")
    partial.replace(destination)
    return destination


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"unsafe archive member: {name}")
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"unsafe archive member: {member.name}")
            bundle.extractall(destination)
    return destination


def read_lmdb(path):
    import lmdb  # pip install lmdb

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        with env.begin() as txn:
            count = pickle.loads(txn.get(b"num_examples"))
            for index in range(count):
                yield pickle.loads(txn.get(str(index).encode()))
    finally:
        env.close()


def plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode()
    return value


def scalar(value):
    value = plain(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected a scalar target")
        value = value[0]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite target")
    return value


def sequence(item, field):
    text = plain(item[field])
    if not isinstance(text, str) or not text:
        raise ValueError(f"invalid sequence field {field}")
    return text.upper()


def convert_item(task, spec, item):
    kind = spec["kind"]
    if kind in ("pair_classification", "pair_regression"):
        features = {"sequence_a": sequence(item, "primary_1"), "sequence_b": sequence(item, "primary_2")}
    else:
        features = {"sequence": sequence(item, "primary")}
    if kind in ("regression", "pair_regression"):
        label = scalar(item[spec["target"]])
    elif kind in ("classification", "pair_classification"):
        label = int(scalar(item[spec["target"]]))
        if not 0 <= label < spec["classes"]:
            raise ValueError(f"{task}: label {label} out of range")
    else:  # ssp
        ss3 = [int(x) for x in plain(item["ss3"])]
        mask = [int(bool(x)) for x in plain(item["valid_mask"])]
        length = len(features["sequence"])
        if len(ss3) != length or len(mask) != length or any(not 0 <= x <= 2 for x in ss3):
            raise ValueError("secondary_structure: label/sequence length mismatch")
        label = {"ss3": ss3, "valid_mask": mask}
    return features, label


def convert_task(task, spec, cache, views):
    archive = download(spec["url"], cache / Path(spec["url"]).name.split("?")[0], spec["md5"])
    extracted = cache / f"extracted-{task}"
    if not (extracted / spec["inner"]).is_dir():
        safe_extract(archive, extracted)
    rows = {"fit": [], "tune": [], "eval": []}
    for source_split, view in spec["splits"].items():
        lmdb_dir = extracted / spec["inner"] / f"{spec['inner']}_{source_split}.lmdb"
        if not lmdb_dir.exists():
            raise FileNotFoundError(f"{task}: missing {lmdb_dir}")
        for index, item in enumerate(read_lmdb(lmdb_dir)):
            features, label = convert_item(task, spec, item)
            rows[view].append({"example_id": f"{task}/{source_split}/{index:06d}",
                               "task": task, "features": features, "label": label})
    for view, expected in spec["counts"].items():
        if len(rows[view]) != expected:
            raise ValueError(f"{task}/{view}: expected {expected} rows, got {len(rows[view])}")
    for view in rows:
        views[view][task] = rows[view]
    print(f"converted {task}: " + ", ".join(f"{v}={len(rows[v])}" for v in ("fit", "tune", "eval")), flush=True)


def dump_jsonl(path, rows, public_only=False, labels_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            if public_only:
                row = {k: row[k] for k in ("example_id", "task", "features")}
            elif labels_only:
                row = {"example_id": row["example_id"], "label": row["label"]}
            out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return sha256(path)


def statistics(views):
    stats = {}
    for task, spec in SPEC.items():
        fit = views["fit"][task]
        if spec["kind"] in ("regression", "pair_regression"):
            values = [row["label"] for row in fit]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            stats[task] = {"kind": "regression", "mean": mean, "std": std}
        elif spec["kind"] == "ssp":
            counts = [0, 0, 0]
            for row in fit:
                for value, valid in zip(row["label"]["ss3"], row["label"]["valid_mask"]):
                    if valid:
                        counts[value] += 1
            stats[task] = {"kind": "ssp", "class_counts": counts}
        else:
            counts = [0] * spec["classes"]
            for row in fit:
                counts[row["label"]] += 1
            stats[task] = {"kind": "classification", "class_counts": counts}
    return stats


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: materialize.py TARGET_DIRECTORY")
    missing = [name for name in ("lmdb", "numpy")
               if __import__("importlib").util.find_spec(name) is None]
    if missing:
        # Checked before the 1.6 GB download rather than failing inside read_lmdb.
        raise SystemExit(f"missing Python packages {missing}: pip install lmdb numpy")
    target = Path(sys.argv[1]).resolve()
    cache = target / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    views = {"fit": {}, "tune": {}, "eval": {}}
    for task, spec in SPEC.items():
        convert_task(task, spec, cache, views)
    stats = statistics(views)

    staging = Path(tempfile.mkdtemp(prefix="peer-materialize-", dir=target))
    development = staging / "public" / "development"
    verifier = staging / "public" / "verifier"
    private = staging / "private"
    development_files, development_counts = {}, {"fit": {}, "tune": {}}
    for view in ("fit", "tune"):
        for task in TASKS:
            name = f"{view}/{task}.jsonl"
            development_files[name] = dump_jsonl(development / name, views[view][task])
            development_counts[view][task] = len(views[view][task])
    write_json(development / "stats.json", stats)
    write_json(development / "manifest.json", {
        "schema": SCHEMA, "view": "development",
        "files": development_files, "counts": development_counts})

    verifier_files = {f"{task}.jsonl": dump_jsonl(verifier / f"{task}.jsonl", views["eval"][task], public_only=True)
                      for task in TASKS}
    write_json(verifier / "manifest.json", {
        "schema": SCHEMA, "view": "verifier", "files": verifier_files,
        "counts": {task: len(views["eval"][task]) for task in TASKS}})

    private_files = {f"{task}.jsonl": dump_jsonl(private / f"{task}.jsonl", views["eval"][task], labels_only=True)
                     for task in TASKS}
    write_json(private / "manifest.json", {
        "schema": PRIVATE_SCHEMA,
        "public_input_manifest_sha256": sha256(verifier / "manifest.json"),
        "files": private_files,
        "fit_standard_deviation": {"ppi_affinity": stats["ppi_affinity"]["std"]},
        "counts": {task: len(views["eval"][task]) for task in TASKS}})

    for name in ("public", "private"):
        final = target / name
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(staging / name), str(final))
    staging.rmdir()
    print(f"materialized {sum(len(v) for view in views.values() for v in view.values())} rows into {target}")


if __name__ == "__main__":
    main()
