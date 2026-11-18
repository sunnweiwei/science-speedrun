"""AWF held-out spatial blind scoring for frozen submissions, using the OlmoEarth Table 2 cosine 20-NN classifier."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

import grade


# Application raster codes -> class indices in the authors' awf_sentinel2 paper task.
# Sources and the three label-vector hashes are documented in tests/README.md.
PAPER_CLASS = (8, 5, 6, 2, 1, 0, 4, 3, 7)
SPLITS = ("train", "valid", "test")
PAPER_COUNTS = {
    "train": (218, 233, 36, 15, 49, 41, 302, 68, 104),
    "valid": (30, 45, 5, 2, 6, 6, 56, 10, 33),
    "test": (40, 42, 8, 1, 4, 8, 54, 12, 31),
}
PAPER_LABEL_SHA = {
    "train": "19e2a9efc70ca3d916ae77d422aaddf3c928ad61a13e55fd9187d0a4cf1b8292",
    "valid": "10d659e3c767eee275033c43ab4071f81a8a4bf481bd9d895efc107f848c7024",
    "test": "383959f1156f498af456da2c198e74b07e2dec0b49bc3d1fd916256b04ac76a3",
}


def aligned_splits(label_rows, roster, dev_splits=None):
    if roster.get("protocol") not in ("olmoearth-table2-awf-s2", "ssi-spatial-awf"):
        raise ValueError("Unknown AWF split protocol")
    if not isinstance(roster.get("source"), str) or not roster["source"].strip():
        raise ValueError("Record the source of the operator's window-to-split mapping")
    labels = {}
    for row in label_rows:
        key, value = row["id"], row["label"]
        if not isinstance(key, str) or key in labels or type(value) not in (float, int) or value not in range(9):
            raise ValueError("Invalid or duplicate SSI label")
        labels[key] = PAPER_CLASS[int(value)]
    rows = roster["rows"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(labels) or set(ids) != labels.keys():
        raise ValueError("Roster must contain each SSI window exactly once")
    by_split = {name: [] for name in SPLITS}
    ordered = all("paper_row" in row for row in rows)
    if not ordered and any("paper_row" in row for row in rows):
        raise ValueError("Partial paper row mapping")
    for row in rows:
        split = row["split"]
        if split not in by_split:
            raise ValueError(f"Unknown split {split!r}")
        by_split[split].append(row)
    result = {}
    for name, members in by_split.items():
        if not members or (name == "train" and len(members) < 20):
            raise ValueError(f"Insufficient {name} windows for a three-way 20-neighbour evaluation")
        if ordered:
            indices = [row["paper_row"] for row in members]
            if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(members))):
                raise ValueError(f"Invalid paper row indices for {name}")
        members.sort(key=lambda row: row["paper_row"] if ordered else row["id"])
        keys = [row["id"] for row in members]
        y = np.array([labels[key] for key in keys], dtype="<i8")
        if roster["protocol"] == "olmoearth-table2-awf-s2":
            if tuple(np.bincount(y, minlength=9)) != PAPER_COUNTS[name]:
                raise ValueError(f"{name} label counts differ from the released paper split")
            if ordered and hashlib.sha256(y.tobytes()).hexdigest() != PAPER_LABEL_SHA[name]:
                raise ValueError(f"{name} label sequence differs from the released paper split")
        result[name] = (keys, y)

    train_ids = set(result["train"][0])
    valid_ids = set(result["valid"][0])
    test_ids = set(result["test"][0])
    if (
        len(train_ids & test_ids) != 0
        or len(valid_ids & test_ids) != 0
        or len(train_ids & valid_ids) != 0
    ):
        raise ValueError("Window-ID overlap between train/valid and test splits")

    if (
        dev_splits is None
        and roster["protocol"] == "ssi-spatial-awf"
        and len(labels) == 1459
    ):
        dev_path = grade.default_dev_splits_path()
        if dev_path is not None:
            dev_splits = json.loads(dev_path.read_text())
    if dev_splits is not None:
        dev_train = set(dev_splits["train"])
        dev_val = set(dev_splits.get("val", dev_splits.get("valid", [])))
        if len(dev_train & test_ids) != 0 or len(dev_val & test_ids) != 0:
            raise ValueError("Development allowlist overlaps held-out spatial test windows")
        if dev_train != train_ids or dev_val != valid_ids:
            raise ValueError("Development allowlist differs from sealed spatial train/valid splits")
    return result


def predict(train, labels, query):
    """The paper's cosine top-20 vote at temperature 0.07, with CPU float32 arithmetic."""
    # Use the upstream torch.topk behaviour, including at tied neighbour distances.
    train = F.normalize(torch.as_tensor(train, dtype=torch.float32), dim=1, eps=1e-8)
    query = F.normalize(torch.as_tensor(query, dtype=torch.float32), dim=1, eps=1e-8)
    similarity, indices = (query @ train.T).topk(20, dim=1)
    votes = F.one_hot(torch.as_tensor(labels)[indices], num_classes=9)
    return (votes * torch.exp(similarity / 0.07).unsqueeze(-1)).sum(1).argmax(1).numpy()


def score_features(path, splits):
    train_ids, valid_ids, test_ids = (set(splits[name][0]) for name in SPLITS)
    if len(train_ids & test_ids) != 0 or len(valid_ids & test_ids) != 0 or len(train_ids & valid_ids) != 0:
        raise ValueError("Window-ID overlap between train/valid and test splits")
    expected = train_ids | valid_ids | test_ids
    with np.load(path, allow_pickle=False) as archive:
        ids, features = archive["ids"], archive["features"]
        if ids.ndim != 1 or ids.dtype.kind not in "US":
            raise ValueError("Expected a vector of window IDs")
        names = ids.astype(str).tolist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("Feature window IDs differ from the roster")
        if (features.ndim != 2 or features.shape[0] != len(names) or features.shape[1] < 1
                or features.dtype.kind != "f" or not np.isfinite(features).all()):
            raise ValueError("Invalid feature matrix")
        index = {key: pos for pos, key in enumerate(names)}
        arrays = {split: features[[index[key] for key in keys]] for split, (keys, _) in splits.items()}
    result = {}
    for name in ("valid", "test"):
        target = splits[name][1]
        predicted = predict(arrays["train"], splits["train"][1], arrays[name])
        correct = int(np.count_nonzero(predicted == target))
        result[name] = {"correct": correct, "n": len(target), "accuracy": correct / len(target),
                        "class_counts": np.bincount(target, minlength=9).tolist()}
    return result


def publish(path, result):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(tmp.name, path)


def grade_package(args, splits, proof):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".awf-knn.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / "reward.json").exists():
            raise ValueError("Use a fresh output directory for the new protocol")
        native = args.native_replay
        if native is None:
            native = Path(tempfile.mkdtemp(prefix="native-", dir=out))
            subprocess.run([sys.executable, str(Path(grade.__file__)), "--inference-only",
                            "--model", str(args.model), "--data", str(args.data),
                            "--probes", str(args.probes), "--output", str(native)], check=True)
            if (native / "reward.json").exists():
                invalid = json.loads((native / "report.json").read_text())
                if invalid.get("status") != "invalid_submission":
                    raise ValueError("Unexpected reward during native inference")
                publish(out / "report.json", {**invalid, **proof})
                publish(out / "reward.json", {"score": 0.0, "reward": 0.0, "invalid_submission": 1})
                return
        else:
            grade.validate_package(args.model)
        manifest = native / "inference-manifest.json"
        grade.validate_manifest(json.loads(manifest.read_text()), args.model, args.data, native, args.probes)
        awf = score_features(native / "awf/features.npz", splits)
        with tempfile.TemporaryDirectory(dir=out) as temp:
            lfmc_file = Path(temp) / "lfmc.json"
            subprocess.run([sys.executable, str(args.probes), "--features", str(native / "lfmc/features.npz"),
                            "--labels", str(args.data / "lfmc/labels.json"), "--task", "lfmc",
                            "--final-test", "--output", str(lfmc_file)], check=True)
            lfmc = json.loads(lfmc_file.read_text())
        mae = lfmc["probe"]["test"]["mae"]
        reward = grade.compute_earth_v4_score(
            awf_test_accuracy=awf["test"]["accuracy"],
            lfmc_test_mae=mae,
            awf_validation_accuracy=awf["valid"]["accuracy"],
        )
        dev_splits_file = grade.default_dev_splits_path()
        report = {
            **reward,
            **proof,
            "status": "scored",
            "awf_dev_splits_sha256": (
                grade.digest(dev_splits_file) if dev_splits_file is not None else None
            ),
            "awf": awf,
            "lfmc": {
                "selected_parameter": lfmc["probe"]["selected_parameter"],
                "validation": lfmc["probe"]["validation"],
                "test": lfmc["probe"]["test"],
                "constant_test": lfmc["constant_test"],
                "frozen_constant_test_mae": grade.LFMC_CONSTANT_TEST_MAE,
            },
            "native_replay": str(native),
            "inference_manifest_sha256": grade.digest(manifest),
            "lfmc_labels_sha256": grade.digest(args.data / "lfmc/labels.json"),
        }
        publish(out / "report.json", report)
        publish(out / "reward.json", reward)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    feature = sub.add_parser("features", help="Regrade independently extracted AWF features")
    feature.add_argument("--features", type=Path, required=True)
    feature.add_argument("--labels", type=Path, required=True)
    package = sub.add_parser("package", help="Run the full SSI verifier with a held-out AWF roster")
    for name in ("model", "data", "probes"):
        package.add_argument("--" + name, type=Path, required=True)
    package.add_argument("--native-replay", type=Path, help="Reuse a native hash-bound inference run")
    for p in (feature, package):
        p.add_argument(
            "--roster",
            type=Path,
            default=grade.default_roster_path(),
            help="Sealed spatial AWF roster (defaults to tests/sealed/spatial-roster.json)",
        )
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    label_path = args.labels if args.mode == "features" else args.data / "awf/labels.json"
    roster_bytes, label_bytes = args.roster.read_bytes(), label_path.read_bytes()
    roster = json.loads(roster_bytes)
    splits = aligned_splits(json.loads(label_bytes), roster)
    proof = {"protocol": roster["protocol"] + "/cosine-20nn-v1", "roster_source": roster["source"],
             "ordered_by": "paper_row" if "paper_row" in roster["rows"][0] else "SSI window ID",
             "split_sizes": {name: len(keys) for name, (keys, _) in splits.items()},
             "roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
             "awf_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
             "scorer_sha256": grade.digest(Path(__file__))}
    if args.mode == "features":
        result = score_features(args.features, splits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        publish(args.output, {**proof, **result, "features_sha256": grade.digest(args.features)})
    else:
        grade_package(args, splits, proof)


if __name__ == "__main__":
    main()
