"""Immutable full or site-sampled validation views; canonical data stays untouched."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_objects(path):
    """Yield raw row objects without deserializing excluded final-test labels."""
    text = Path(path).read_text()
    if not text.lstrip().startswith("[") or not text.rstrip().endswith("]"):
        raise ValueError("Expected a JSON array of release rows")
    depth, quoted, escaped, start = 0, False, False, None
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed release rows")
            if depth == 0:
                yield text[start : index + 1]
    if depth or quoted:
        raise ValueError("Malformed release rows")


def trainval_rows(path):
    rows = []
    for index, raw in enumerate(json_objects(path)):
        # The release's split field precedes labels. Check it before decoding a
        # row, so test labels are never turned into values or passed to probes.
        split = re.search(r'"split"\s*:\s*"(train|val|test)"', raw)
        if split is None:
            raise ValueError("Missing official split")
        if split[1] == "test":
            continue
        row = json.loads(raw)
        if row["split"] not in ("train", "val") or row["index"] != index:
            raise ValueError("Unexpected release row order or split")
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate train/validation IDs")
    return rows


def site(row):
    # Same LFMC site definition as compare_seeds.py, fixed before fitting.
    return tuple(np.round(np.asarray(row["latlon"], dtype=float), 6).tolist())


def sample_lfmc(rows, seed, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Site fraction must be in (0, 1]")
    sites = sorted({site(row) for row in rows})
    ranked = sorted(
        sites,
        key=lambda value: hashlib.sha256(
            json.dumps([seed, value], separators=(",", ":")).encode()
        ).digest(),
    )
    selected = set(ranked[: math.ceil(len(sites) * fraction)])
    subset = [row for row in rows if site(row) in selected]
    if {row["split"] for row in subset} != {"train", "val"}:
        raise ValueError(
            "Site sample omitted train or validation; choose its policy before scoring"
        )
    return subset, sorted(selected), len(sites)


def copy_inputs(source_path, destination, rows, task):
    """Copy selected source rows byte-for-byte, with their canonical ordering."""
    indices = [row["index"] for row in rows]
    with h5py.File(source_path, "r") as source, h5py.File(destination, "x") as dest:
        if source.attrs["task"] != task:
            raise ValueError("Wrong input task")
        if source["ids"].asstr()[indices].tolist() != [row["id"] for row in rows]:
            raise ValueError("Input IDs disagree with canonical row indices")
        dest.attrs.update(source.attrs)
        for name, values in source.items():
            shape = (len(indices), *values.shape[1:])
            options = {}
            if values.chunks:
                options["chunks"] = (
                    min(values.chunks[0], len(indices)),
                    *values.chunks[1:],
                )
            if values.compression:
                options["compression"] = values.compression
                if values.compression_opts is not None:
                    options["compression_opts"] = values.compression_opts
            target = dest.create_dataset(
                name, shape=shape, dtype=values.dtype, **options
            )
            target.attrs.update(values.attrs)
            for start in range(0, len(indices), 64):
                target[start : start + 64] = values[indices[start : start + 64]]


def prepare(data_root, output, seed=20260930, fraction=0.125):
    source_root = (Path(data_root) / "eval").resolve()
    output = Path(output).resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Screening view must be outside canonical evaluation data")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a fixed screening population")
    selections = {}
    for task in ("awf", "lfmc"):
        rows = trainval_rows(source_root / task / "labels.json")
        if task == "lfmc":
            rows, sites, total_sites = sample_lfmc(rows, seed, fraction)
        else:
            sites, total_sites = None, None
        selections[task] = (rows, sites, total_sites)
    output.mkdir(parents=True)
    manifest = {
        "kind": "fixed_validation_view" if fraction == 1 else "fixed_screening_view",
        "schema_version": 1,
        "population": (
            "full AWF and LFMC train/val"
            if fraction == 1
            else "full AWF train/val; LFMC site-sampled train/val"
        ),
        "selection_seed": seed,
        "lfmc_site_fraction": fraction,
        "selection_method": "lowest SHA256 of compact JSON [seed, rounded latlon]; ceil(fraction * sites)",
        "site_coordinates_decimals": 6,
        "canonical_data_immutable": True,
        "test_rows_in_view": 0,
        "test_labels_deserialized": False,
        "inference": {"batch_size": 16, "python_hash_seed": 0},
        "probe_protocol": "unchanged environment/evaluation/probes.py grids",
        "tasks": {},
    }
    for task, (rows, sites, total_sites) in selections.items():
        folder = output / task
        folder.mkdir()
        (folder / "labels.json").write_text(json.dumps(rows, indent=2) + "\n")
        copy_inputs(source_root / task / "inputs.h5", folder / "inputs.h5", rows, task)
        membership = [
            {"id": row["id"], "index": row["index"], "split": row["split"]}
            for row in rows
        ]
        manifest["tasks"][task] = {
            "source": str(source_root / task),
            "rows": dict(Counter(row["split"] for row in rows)),
            "canonical_indices_and_ids": membership,
            "ordered_ids_sha256": hashlib.sha256(
                json.dumps([row["id"] for row in rows]).encode()
            ).hexdigest(),
            "input_sha256": sha256(folder / "inputs.h5"),
            "labels_sha256": sha256(folder / "labels.json"),
            "selected_sites": sites,
            "canonical_trainval_site_count": total_sites,
        }
        for path in folder.iterdir():
            path.chmod(0o444)
        folder.chmod(0o555)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    path.chmod(0o444)
    output.chmod(0o555)
    return manifest


def check_cache(
    random_package, candidate_package, cached_context, current_context, task
):
    """Reject historical/full-view or different-architecture random caches."""
    packages = [Path(random_package), Path(candidate_package)]
    provenance = [
        json.loads((path / "provenance.json").read_text()) for path in packages
    ]
    if (
        provenance[0].get("origin") != "random_initialization"
        or provenance[0].get("checkpoint") is not None
    ):
        raise ValueError("Cached control must be an untrained random encoder")
    if provenance[1].get("origin") != "own_from_scratch_training":
        raise ValueError("Candidate must be our from-scratch trained encoder")
    if json.loads((packages[0] / "config.json").read_text()) != json.loads(
        (packages[1] / "config.json").read_text()
    ):
        raise ValueError("Cached control architecture differs")
    for key in (
        "upstream",
        "initialization_seed",
        "parameters",
        "inference_adapter_version",
        "infer_sha256",
    ):
        if key not in provenance[0] or provenance[0][key] != provenance[1].get(key):
            raise ValueError(f"Cached control differs in {key}")
    for path, info in zip(packages, provenance):
        if (
            sha256(path / "infer.py") != info["infer_sha256"]
            or sha256(path / "encoder.pt") != info["encoder_sha256"]
        ):
            raise ValueError("Package files disagree with their provenance")
    if provenance[1].get("training_validity") == "invalid":
        raise ValueError("Known invalid candidate training")
    for key in ("gpu_type", "torch", "cuda", "batch_size", "python_hash_seed"):
        if key not in cached_context or cached_context[key] != current_context.get(key):
            raise ValueError(f"Cached inference context differs in {key}")
    if current_context["batch_size"] != 16 or current_context["python_hash_seed"] != 0:
        raise ValueError("Screening requires inference batch 16 and hash seed 0")
    cached_input = cached_context.get("input_sha256", {}).get(task)
    if not cached_input or cached_input != current_context.get("input_sha256", {}).get(
        task
    ):
        raise ValueError(
            "Cached control has a different or unrecorded input view/batch ordering"
        )
    return {"compatible": True, "task": task, "input_sha256": cached_input}


def population(view):
    view = Path(view).resolve()
    manifest = json.loads((view / "manifest.json").read_text())
    if (
        manifest.get("kind") not in ("fixed_screening_view", "fixed_validation_view")
        or manifest.get("test_rows_in_view") != 0
    ):
        raise ValueError("Expected fixed train/validation view")
    return {
        "kind": (
            "full_validation"
            if manifest["kind"] == "fixed_validation_view"
            else "screening"
        ),
        "manifest": str(view / "manifest.json"),
        "manifest_sha256": sha256(view / "manifest.json"),
        "description": manifest["population"],
        "final_recipe_criterion_established": False,
    }


def annotate(path, view):
    report = json.loads(Path(path).read_text())
    report["evaluation_population"] = population(view)
    Path(path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=20260930)
    build.add_argument("--site-fraction", type=float, default=0.125)
    cache = commands.add_parser("check-cache")
    cache.add_argument("--random-package", type=Path, required=True)
    cache.add_argument("--candidate-package", type=Path, required=True)
    cache.add_argument("--cached-context", type=Path, required=True)
    cache.add_argument("--current-context", type=Path, required=True)
    cache.add_argument("--task", choices=["awf", "lfmc"], required=True)
    probe = commands.add_parser(
        "probe", help="Unchanged probe grids with explicitly labelled screening reports"
    )
    probe.add_argument("--view", type=Path, required=True)
    probe.add_argument("--features", type=Path, required=True)
    probe.add_argument("--task", choices=["awf", "lfmc"], required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--validation-predictions", type=Path, required=True)
    compare = commands.add_parser(
        "compare", help="Existing paired bootstrap on the same screening view"
    )
    compare.add_argument("--view", type=Path, required=True)
    compare.add_argument("--data-root", type=Path, required=True)
    compare.add_argument("--pairs", nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--raw-baseline", help="Optional raw-feature report name; reported separately"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.data_root, args.output, args.seed, args.site_fraction)
        print(
            json.dumps(
                {
                    "view": str(args.output.resolve()),
                    "manifest_sha256": sha256(args.output / "manifest.json"),
                    "tasks": {
                        task: {
                            "rows": info["rows"],
                            "input_sha256": info["input_sha256"],
                            "selected_sites": (
                                len(info["selected_sites"])
                                if info["selected_sites"] is not None
                                else None
                            ),
                        }
                        for task, info in manifest["tasks"].items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "check-cache":
        print(
            json.dumps(
                check_cache(
                    args.random_package,
                    args.candidate_package,
                    json.loads(args.cached_context.read_text()),
                    json.loads(args.current_context.read_text()),
                    args.task,
                ),
                indent=2,
            )
        )
    else:
        population(args.view)
        here = Path(__file__).resolve().parent
        track = here.parent
        if args.command == "probe":
            info = json.loads((args.view / "manifest.json").read_text())["tasks"][
                args.task
            ]
            labels = args.view / args.task / "labels.json"
            if sha256(labels) != info["labels_sha256"]:
                raise ValueError("Fixed screening labels changed")
            command = [
                sys.executable,
                str(
                    here / "probes.py"
                    if (here / "probes.py").is_file()
                    else track / "environment/evaluation/probes.py"
                ),
                "--features",
                str(args.features),
                "--labels",
                str(labels),
                "--task",
                args.task,
                "--validation-predictions",
                str(args.validation_predictions),
                "--output",
                str(args.output),
            ]
        else:
            if (args.data_root / "eval").resolve() != args.view.resolve():
                raise ValueError(
                    "Comparison data root must point to the fixed screening view"
                )
            command = [
                sys.executable,
                str(here / "compare_seeds.py"),
                "--data-root",
                str(args.data_root),
                "--pairs",
                *args.pairs,
                "--output",
                str(args.output),
            ]
            if args.raw_baseline:
                command.extend(["--raw-baseline", args.raw_baseline])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        print(json.dumps(annotate(args.output, args.view), indent=2))


if __name__ == "__main__":
    main()
