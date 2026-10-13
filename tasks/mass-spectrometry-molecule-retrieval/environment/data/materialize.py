#!/usr/bin/env python3
"""Create label-separated MassSpecGym retrieval views from pinned v1.5 files."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


DATASET_RELATIVE = Path("data/MassSpecGym1.5.tsv")
CANDIDATES_RELATIVE = Path(
    "data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json"
)
DATASET_SHA256 = "50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a"
CANDIDATES_SHA256 = "9536a330cba24c399271364889119f90124579400dba14e9f78630c75c13c73a"
HF_REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"
REQUIRED_COLUMNS = {
    "identifier", "mzs", "intensities", "smiles", "inchikey", "formula",
    "precursor_mz", "adduct", "instrument_type", "collision_energy", "fold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value) -> None:
    path.write_text(compact_json(value) + "\n", encoding="utf-8")


def write_row(stream, value) -> None:
    stream.write(compact_json(value) + "\n")


def candidate_id(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def parse_float(value: str, *, nullable: bool = False):
    if nullable and (value == "" or value.lower() == "nan"):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number: {value!r}")
    return number


def parse_spectrum(row: dict) -> dict:
    mzs = [parse_float(value) for value in row["mzs"].split(",")]
    intensities = [parse_float(value) for value in row["intensities"].split(",")]
    if not mzs or len(mzs) != len(intensities):
        raise ValueError(f"invalid peak arrays for {row['identifier']}")
    if any(mz <= 0 for mz in mzs) or mzs != sorted(mzs):
        raise ValueError(f"invalid or unsorted m/z values for {row['identifier']}")
    if any(value < 0 for value in intensities) or max(intensities) <= 0:
        raise ValueError(f"invalid intensities for {row['identifier']}")
    precursor_mz = parse_float(row["precursor_mz"])
    if precursor_mz <= 0:
        raise ValueError(f"invalid precursor m/z for {row['identifier']}")
    return {
        "query_id": row["identifier"],
        "mzs": mzs,
        "intensities": intensities,
        "precursor_mz": precursor_mz,
        "adduct": row["adduct"],
        "instrument_type": row["instrument_type"],
        "collision_energy": parse_float(row["collision_energy"], nullable=True),
    }


def make_pool(candidates: list[str]) -> tuple[str, list[dict]]:
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("candidate pools must contain between 1 and 256 entries")
    if any(not isinstance(smiles, str) or not smiles for smiles in candidates):
        raise ValueError("candidate SMILES must be nonempty strings")
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate candidate SMILES")
    identified = [(candidate_id(smiles), smiles) for smiles in candidates]
    if len({item[0] for item in identified}) != len(identified):
        raise ValueError("candidate SHA-256 collision")
    identity_payload = "massspecgym-candidate-pool/v1\n" + "\n".join(
        sorted(item[0] for item in identified)
    )
    pool_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    identified.sort(
        key=lambda item: hashlib.sha256(
            (pool_id + "\0" + item[0]).encode("ascii")
        ).hexdigest()
    )
    return pool_id, [
        {"candidate_id": identifier, "smiles": smiles}
        for identifier, smiles in identified
    ]


def with_target(public: dict, source: dict) -> dict:
    return {
        **public,
        "target_smiles": source["smiles"],
        "target_inchikey": source["inchikey"],
        "target_formula": source["formula"],
    }


def manifest(
    root: Path,
    schema: str,
    counts: dict,
    names: list[str],
    *,
    dataset_sha256: str,
    candidates_sha256: str,
) -> dict:
    return {
        "schema": schema,
        "source": {
            "repo_id": "roman-bushuiev/MassSpecGym",
            "revision": HF_REVISION,
            "dataset_sha256": dataset_sha256,
            "candidates_sha256": candidates_sha256,
        },
        "counts": counts,
        "files": {name: sha256(root / name) for name in names},
    }


def assert_new_destination(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value


def validate_manifest_files(root: Path, files: dict) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError(f"manifest has no file digests: {root}")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe manifest filename in {root}: {name!r}")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe materialized file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def validate_existing_outputs(
    output_root: Path,
    private_root: Path,
    *,
    dataset_sha256: str,
    candidates_sha256: str,
) -> dict:
    """Validate and reuse a complete output pair; never repair it in place."""
    development_root = output_root / "development"
    verifier_root = output_root / "verifier"
    development = read_json(development_root / "manifest.json")
    verifier = read_json(verifier_root / "manifest.json")
    private = read_json(private_root / "manifest.json")

    expected_source = {
        "repo_id": "roman-bushuiev/MassSpecGym",
        "revision": HF_REVISION,
        "dataset_sha256": dataset_sha256,
        "candidates_sha256": candidates_sha256,
    }
    for name, value, expected_schema in (
        ("development", development, "massspecgym-retrieval-development/v1"),
        ("verifier", verifier, "massspecgym-retrieval-input/v1"),
    ):
        if value.get("schema") != expected_schema:
            raise ValueError(f"unexpected {name} manifest schema")
        if value.get("source") != expected_source:
            raise ValueError(f"unexpected {name} source provenance")
    if private.get("schema") != "massspecgym-retrieval-private-labels/v1":
        raise ValueError("unexpected private-label manifest schema")
    if private.get("source_revision") != HF_REVISION:
        raise ValueError("unexpected private-label source revision")

    validate_manifest_files(development_root, development.get("files"))
    validate_manifest_files(verifier_root, verifier.get("files"))
    validate_manifest_files(private_root, private.get("files"))

    counts = development.get("counts")
    verifier_counts = verifier.get("counts")
    if not isinstance(counts, dict) or not isinstance(verifier_counts, dict):
        raise ValueError("invalid materialized counts")
    if counts.get("test_queries") != verifier_counts.get("queries"):
        raise ValueError("public development/verifier query-count mismatch")
    if counts.get("test_candidate_pools") != verifier_counts.get("candidate_pools"):
        raise ValueError("public development/verifier pool-count mismatch")
    if counts.get("test_queries") != private.get("queries"):
        raise ValueError("public/private query-count mismatch")
    return counts


def materialize(args) -> dict:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    private_root = args.private_label_root.resolve()
    dataset_path = source_root / DATASET_RELATIVE
    candidates_path = source_root / CANDIDATES_RELATIVE
    for path, expected in (
        (dataset_path, args.dataset_sha256),
        (candidates_path, args.candidates_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    if (
        output_root == private_root
        or output_root.is_relative_to(private_root)
        or private_root.is_relative_to(output_root)
    ):
        raise ValueError("public and private roots must be separate, non-nested paths")
    reuse_existing = getattr(args, "reuse_valid_existing", False)
    if output_root.exists() or private_root.exists():
        if not reuse_existing:
            existing = output_root if output_root.exists() else private_root
            raise FileExistsError(f"refusing to overwrite existing output: {existing}")
        if not output_root.is_dir() or not private_root.is_dir():
            raise ValueError("cannot reuse a partial or non-directory output pair")
        return validate_existing_outputs(
            output_root,
            private_root,
            dataset_sha256=args.dataset_sha256,
            candidates_sha256=args.candidates_sha256,
        )
    assert_new_destination(output_root)
    assert_new_destination(private_root)

    public_temp = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp.", dir=output_root.parent))
    private_temp = Path(tempfile.mkdtemp(prefix=private_root.name + ".tmp.", dir=private_root.parent))
    development = public_temp / "development"
    verifier = public_temp / "verifier"
    development.mkdir()
    verifier.mkdir()
    validation_rows, test_rows = [], []
    identifiers = set()
    unique_structures = {fold: set() for fold in ("train", "val", "test")}
    fold_counts = {fold: 0 for fold in ("train", "val", "test")}

    try:
        with dataset_path.open(encoding="utf-8", newline="") as source, (
            development / "train.jsonl"
        ).open("w", encoding="utf-8") as train_stream:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
                raise ValueError("MassSpecGym1.5.tsv is missing required columns")
            for source_row in reader:
                fold = source_row["fold"]
                if fold not in fold_counts:
                    raise ValueError(f"unknown fold: {fold}")
                query_id = source_row["identifier"]
                if not query_id or query_id in identifiers:
                    raise ValueError(f"missing or duplicate identifier: {query_id!r}")
                identifiers.add(query_id)
                if not source_row["smiles"] or not source_row["inchikey"]:
                    raise ValueError(f"missing molecular label for {query_id}")
                public_row = parse_spectrum(source_row)
                fold_counts[fold] += 1
                unique_structures[fold].add(source_row["inchikey"])
                if fold == "train":
                    write_row(train_stream, with_target(public_row, source_row))
                elif fold == "val":
                    validation_rows.append((public_row, source_row))
                else:
                    test_rows.append((public_row, source_row))

        if any(unique_structures[left] & unique_structures[right] for left, right in (
            ("train", "val"), ("train", "test"), ("val", "test")
        )):
            raise ValueError("2D structure leakage across folds")

        with candidates_path.open(encoding="utf-8") as stream:
            candidates_by_truth = json.load(stream)
        if not isinstance(candidates_by_truth, dict):
            raise ValueError("candidate source must be a JSON object")

        validation_pools, test_pools = {}, {}
        with (development / "validation.jsonl").open("w", encoding="utf-8") as val_stream, (
            verifier / "queries.jsonl"
        ).open("w", encoding="utf-8") as test_stream, (
            private_temp / "labels.jsonl"
        ).open("w", encoding="utf-8") as label_stream:
            for fold, rows, pools, public_stream in (
                ("validation", validation_rows, validation_pools, val_stream),
                ("test", test_rows, test_pools, test_stream),
            ):
                for public_row, source_row in rows:
                    truth = source_row["smiles"]
                    if truth not in candidates_by_truth:
                        raise ValueError(f"no candidates for {public_row['query_id']}")
                    candidates = candidates_by_truth[truth]
                    if candidates.count(truth) != 1:
                        raise ValueError(
                            f"truth must occur exactly once in candidate pool: {public_row['query_id']}"
                        )
                    pool_id, pool_candidates = make_pool(candidates)
                    previous = pools.setdefault(pool_id, pool_candidates)
                    if previous != pool_candidates:
                        raise ValueError("candidate pool hash collision")
                    query = {**public_row, "candidate_pool_id": pool_id}
                    if fold == "validation":
                        write_row(public_stream, with_target(query, source_row))
                    else:
                        write_row(public_stream, query)
                        write_row(label_stream, {
                            "query_id": public_row["query_id"],
                            "correct_candidate_id": candidate_id(truth),
                            "true_smiles": truth,
                        })

        def write_pools(path: Path, pools: dict) -> None:
            with path.open("w", encoding="utf-8") as stream:
                for pool_id in sorted(pools):
                    write_row(stream, {
                        "candidate_pool_id": pool_id,
                        "candidates": pools[pool_id],
                    })

        write_pools(development / "validation_candidate_pools.jsonl", validation_pools)
        write_pools(verifier / "candidate_pools.jsonl", test_pools)

        public_counts = {
            "train_queries": fold_counts["train"],
            "validation_queries": fold_counts["val"],
            "validation_candidate_pools": len(validation_pools),
            "test_queries": fold_counts["test"],
            "test_candidate_pools": len(test_pools),
            "unique_structures": {fold: len(values) for fold, values in unique_structures.items()},
        }
        write_json(development / "manifest.json", manifest(
            development,
            "massspecgym-retrieval-development/v1",
            public_counts,
            ["train.jsonl", "validation.jsonl", "validation_candidate_pools.jsonl"],
            dataset_sha256=args.dataset_sha256,
            candidates_sha256=args.candidates_sha256,
        ))
        write_json(verifier / "manifest.json", manifest(
            verifier,
            "massspecgym-retrieval-input/v1",
            {
                "queries": fold_counts["test"],
                "candidate_pools": len(test_pools),
            },
            ["queries.jsonl", "candidate_pools.jsonl"],
            dataset_sha256=args.dataset_sha256,
            candidates_sha256=args.candidates_sha256,
        ))
        write_json(private_temp / "manifest.json", {
            "schema": "massspecgym-retrieval-private-labels/v1",
            "source_revision": HF_REVISION,
            "queries": fold_counts["test"],
            "files": {"labels.jsonl": sha256(private_temp / "labels.jsonl")},
        })

        for directory in (public_temp, development, verifier):
            directory.chmod(0o755)
        for path in public_temp.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        private_temp.chmod(0o700)
        for path in private_temp.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        os.replace(public_temp, output_root)
        os.replace(private_temp, private_root)
        return public_counts
    except Exception:
        shutil.rmtree(public_temp, ignore_errors=True)
        shutil.rmtree(private_temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--private-label-root", required=True, type=Path)
    parser.add_argument("--dataset-sha256", default=DATASET_SHA256)
    parser.add_argument("--candidates-sha256", default=CANDIDATES_SHA256)
    parser.add_argument(
        "--reuse-valid-existing",
        action="store_true",
        help="reuse an existing public/private pair only after manifest validation",
    )
    args = parser.parse_args()
    report = materialize(args)
    print(compact_json(report))


if __name__ == "__main__":
    main()
