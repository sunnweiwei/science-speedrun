"""Replay an inference package and score it with the pinned native FoldBench code."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import stat
import struct
import subprocess
import sys
import tempfile
import time
import traceback


TRACKS = (
    "monomer_protein", "monomer_rna", "monomer_dna",
    "interface_protein_protein", "interface_antibody_antigen",
    "interface_protein_peptide", "interface_protein_rna",
    "interface_protein_dna", "interface_protein_ligand",
)
FOLD_GROUPS = {
    "Protein monomers": ("monomer_protein",),
    "Nucleic-acid monomers": ("monomer_rna", "monomer_dna"),
    "Protein interfaces": (
        "interface_protein_protein",
        "interface_antibody_antigen",
        "interface_protein_peptide",
    ),
    "Protein\u2013nucleic acid": ("interface_protein_rna", "interface_protein_dna"),
    "Protein\u2013ligand": ("interface_protein_ligand",),
}
NA_TRACKS = {"interface_protein_rna", "interface_protein_dna"}
FIELDS = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"]
FEATURE_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"
FEATURE_WIDTH = 2560
FEATURE_GROUP_SIZE = 64
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
# environment/contract.md "Resources": the whole batch must finish within this wall time.
INFERENCE_SECONDS = 8 * 3600
GRADER_WAIT_SECONDS = INFERENCE_SECONDS + 900
SEALED_SCHEMA = "foldbench-sealed-predictions/v1"
FEATURE_BOUNDARY = "ESMCModel.transformer.final_layer_norm"
KERNEL_BACKEND = "pytorch-sdpa-no-xformers-no-transformer-engine"
FORBIDDEN_WEIGHT_SHA256 = {
    # biohub/ESMFold2-Fast b28d8ace
    "60ca19f2898188beba92944365f7b909efd9c99212f5018af75cc47cd9a6184a",
    # biohub/ESMC-6B 45b0fa5d, shards 1..6
    "bd90149ff223e6ac1a0cac6147a5ae0df20d3a21df4f65356a1f19cd14f4aa8a",
    "f75e2144d8269fe2eb4b3e0823fb089b94f176d8024153e85b8fb573a42294fa",
    "f699f01ecc9691d9c6470492765fe54b8b5d2e9f277c139e89427433ffdfe0b2",
    "46add1b7be098bbfdc3073884851ba3057f1b33ea23a158b650a37007dabd13d",
    "1e1cb62f060a34e18f54a31a76683ef888b8cec59e73315f5b31d25d45a1f88c",
    "56c73e13ae96e777ce65eee99364056069ef93b646470f352f83c5f1037b1b18",
}


class InvalidSubmission(Exception):
    """The delivered package is invalid or its setup/inference failed: scored 0, not an infrastructure error."""


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def contained_file(root, relative, *, require_file=True):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Expected a relative file path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or (require_file and not path.is_file()):
        raise ValueError(f"Missing or escaping file: {relative}")
    return path


def validate_no_symlinks_or_escapes(directory: Path):
    """Ensure `directory` and every entry inside it contain zero symlinks or path escapes."""
    if os.path.islink(directory):
        raise InvalidSubmission(f"Output/sealed root must not be a symlink: {directory}")
    if not directory.exists():
        raise InvalidSubmission(f"Missing output/sealed directory: {directory}")
    resolved_root = directory.resolve()
    for current_dir, subdirs, filenames in os.walk(directory, followlinks=False):
        here = Path(current_dir)
        if os.path.islink(here):
            raise InvalidSubmission(f"Symlink directory forbidden in predictions: {here}")
        if not here.resolve().is_relative_to(resolved_root):
            raise InvalidSubmission(f"Directory escapes output root: {here}")
        for name in [*subdirs, *filenames]:
            entry = here / name
            if os.path.islink(entry):
                raise InvalidSubmission(
                    f"Symlink forbidden in predictions/sealed output: {entry.relative_to(directory)}"
                )
            info = os.lstat(entry)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise InvalidSubmission(
                    f"Special file forbidden in predictions/sealed output: {entry.relative_to(directory)}"
                )
            if not entry.resolve().is_relative_to(resolved_root):
                raise InvalidSubmission(
                    f"Path escapes predictions/sealed output: {entry.relative_to(directory)}"
                )


def suite_requests(suite):
    manifest = json.loads((suite / "suite_manifest.json").read_text())
    if manifest.get("contains_ground_truth_coordinates") is not False:
        raise ValueError("Expected a coordinate-free input suite")
    requests = {}
    for record in manifest["requests"]:
        request_id = record["request_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id) or request_id in requests:
            raise ValueError("Unsafe or duplicate suite request ID")
        path = contained_file(suite, record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Suite request checksum mismatch: {request_id}")
        request = json.loads(path.read_text())
        if set(request) != {"schema_version", "request_id", "target", "run"}:
            raise ValueError("Unexpected inference-request fields")
        if request["request_id"] != request_id or request["schema_version"] != "foldbench-folding-request/v0":
            raise ValueError("Suite and request schema/ID differ")
        seeds, samples = request["run"]["seeds"], request["run"]["num_samples"]
        if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(seeds) != len(set(seeds)) or type(samples) is not int or samples < 1):
            raise ValueError("Invalid seed/sample policy")
        requests[request_id] = request
    if not requests:
        raise ValueError("Empty input suite")
    return requests


def suite_targets(suite, requests, ground_truth=None):
    targets = {track: read_csv(suite / "targets" / f"{track}.csv")
               for track in TRACKS if (suite / "targets" / f"{track}.csv").is_file()}
    targets = {track: rows for track, rows in targets.items() if rows}
    if not targets:
        raise ValueError("No scoring targets")
    for rows in targets.values():
        for row in rows:
            if row["pdb_id"] not in requests:
                raise ValueError("Target table references an unknown request")
            if ground_truth is not None and not (ground_truth / (row["pdb_id"] + ".cif")).is_file():
                raise FileNotFoundError(f"Missing native structure: {row['pdb_id']}")
    return targets


def required_protein_features(requests):
    required = {}
    for request in requests.values():
        for entity in request["target"]["entities"]:
            if entity["type"] != "protein":
                continue
            sequence = entity["sequence"]
            digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            previous = required.setdefault(digest, len(sequence))
            if previous != len(sequence):
                raise ValueError("Sequence digest collision")
    return required


def safetensors_header(path):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > min(16 * 1024 * 1024, file_size - 8):
            raise ValueError(f"Invalid safetensors header length: {path}")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    return header, file_size - 8 - header_size


def validate_feature_payload(path, digest, length):
    header, payload_size = safetensors_header(path)
    metadata = header.pop("__metadata__", {})
    if set(header) != {"values", "scales"} or not isinstance(metadata, dict):
        raise ValueError(f"Invalid canonical feature tensors: {digest}")
    expected = {
        "values": ("I8", [length, FEATURE_WIDTH], length * FEATURE_WIDTH),
        "scales": (
            "BF16", [length, FEATURE_WIDTH // FEATURE_GROUP_SIZE],
            length * (FEATURE_WIDTH // FEATURE_GROUP_SIZE) * 2,
        ),
    }
    segments = []
    for name, (dtype, shape, byte_count) in expected.items():
        item = header.get(name)
        offsets = item.get("data_offsets") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or item.get("dtype") != dtype
                or item.get("shape") != shape or not isinstance(offsets, list)
                or len(offsets) != 2 or not all(isinstance(value, int) for value in offsets)
                or offsets[1] - offsets[0] != byte_count):
            raise ValueError(f"Invalid canonical feature tensor {name}: {digest}")
        segments.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(segments):
        if start != cursor or end < start:
            raise ValueError(f"Noncontiguous canonical feature payload: {digest}")
        cursor = end
    if cursor != payload_size:
        raise ValueError(f"Canonical feature payload length mismatch: {digest}")
    required_metadata = {
        "sequence_sha256": digest,
        "esmc_repo": "biohub/ESMC-6B",
        "esmc_revision": ESMC_REVISION,
        "boundary": FEATURE_BOUNDARY,
        "kernel_backend": KERNEL_BACKEND,
        "schema_version": FEATURE_VERSION,
        "quantization": "symmetric-int8",
        "group_size": str(FEATURE_GROUP_SIZE),
        "scale_dtype": "bfloat16",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise ValueError(f"Invalid canonical feature provenance: {digest}")


def suite_features(manifest, requests):
    required = required_protein_features(requests)
    records = {}
    with manifest.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            digest = record.get("sequence_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"Invalid feature digest at line {line_number}")
            if digest in records:
                raise ValueError(f"Duplicate feature digest: {digest}")
            length = record.get("length")
            if (record.get("feature_version") != FEATURE_VERSION
                    or record.get("dtype") != "int8"
                    or record.get("scale_dtype") != "bfloat16"
                    or record.get("quantization") != "symmetric-int8"
                    or record.get("group_size") != FEATURE_GROUP_SIZE
                    or record.get("shape") != [length, FEATURE_WIDTH]
                    or record.get("scale_shape")
                    != [length, FEATURE_WIDTH // FEATURE_GROUP_SIZE]):
                raise ValueError(f"Invalid feature metadata for {digest}")
            path = contained_file(manifest.parent, record.get("path", ""))
            validate_feature_payload(path, digest, length)
            records[digest] = dict(record, source_path=path)
    if set(records) != set(required):
        missing = sorted(set(required) - set(records))
        extra = sorted(set(records) - set(required))
        raise ValueError(
            f"Feature coverage mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    for digest, length in required.items():
        if records[digest]["length"] != length:
            raise ValueError(f"Feature length mismatch for {digest}")
    return records


def write_inference_features(inputs, records):
    feature_dir = inputs / "features/esmc6b-final-int8"
    feature_dir.mkdir(parents=True)
    with (inputs / "features.esmc6b_final_int8.jsonl").open("w") as stream:
        for digest in sorted(records):
            record = records[digest]
            relative = Path("features/esmc6b-final-int8") / f"{digest}.safetensors"
            os.symlink(record["source_path"], inputs / relative)
            public = {key: record[key] for key in (
                "sequence_sha256", "length", "shape", "dtype", "scale_shape",
                "scale_dtype", "quantization", "group_size", "feature_version"
            )}
            public["path"] = relative.as_posix()
            stream.write(json.dumps(public, sort_keys=True) + "\n")


def audit_submission(submission):
    for path in submission.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(submission):
            raise ValueError("Submission symlink escapes the delivered directory")
        relative_parts = path.relative_to(submission).parts
        if any(re.fullmatch(r"candidate[-_]?v?\d+", part, re.IGNORECASE)
               for part in relative_parts):
            raise ValueError("Submission contains a historical candidate directory")
        if path.is_file() and re.search(
            r"(^|[-_.])(optimizer|rollback|superseded|old)([-_.]|$)",
            path.name,
            re.IGNORECASE,
        ):
            raise ValueError("Submission contains training or historical state")
        if not path.is_file() or path.suffix.lower() not in {
            ".safetensors", ".bin", ".pt", ".pth", ".ckpt"
        }:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() in FORBIDDEN_WEIGHT_SHA256:
            raise ValueError(
                f"Submission contains a forbidden supplied checkpoint: {path.name}"
            )


def validate_mmcif(path):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif = MMCIF2Dict(io.StringIO(path.read_text()))
    keys = ("label_atom_id", "label_comp_id", "label_asym_id", "Cartn_x", "Cartn_y", "Cartn_z")
    columns = [cif.get("_atom_site." + key, []) for key in keys]
    if any(not isinstance(column, list) or not column for column in columns):
        raise ValueError("Missing nonempty atom_site columns")
    if len({len(column) for column in columns}) != 1:
        raise ValueError("Atom-site columns have different lengths")
    if any(not math.isfinite(float(value)) for column in columns[-3:] for value in column):
        raise ValueError("Nonfinite atomic coordinates")


def predictions(output, requests):
    accepted, invalid, seen = [], [], set()
    ref_csv = output / "prediction_reference.csv"
    if os.path.islink(ref_csv) or not ref_csv.is_file():
        raise ValueError("Missing or symlinked prediction_reference.csv")
    with ref_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Prediction CSV must have columns {FIELDS}")
        for row in reader:
            request = requests.get(row["pdb_id"])
            seed, sample = int(row["seed"]), int(row["sample"])
            key = (row["pdb_id"], seed, sample)
            if (request is None or seed not in request["run"]["seeds"]
                    or not 0 <= sample < request["run"]["num_samples"] or key in seen):
                raise ValueError(f"Duplicate or unrequested prediction: {key}")
            seen.add(key)
            ranking = float(row["ranking_score"])
            if not math.isfinite(ranking):
                raise ValueError("Nonfinite ranking score")
            raw_rel = Path(row["prediction_path"])
            if raw_rel.is_absolute():
                raise ValueError("Expected a relative file path")
            unlinked_candidate = output / raw_rel
            if os.path.islink(unlinked_candidate):
                raise ValueError(f"Symlink prediction file forbidden: {row['prediction_path']}")
            # A path escape is a contract violation, not an ordinary bad structure.
            path = contained_file(output, row["prediction_path"], require_file=False)
            try:
                if path.suffix.lower() != ".cif":
                    raise ValueError("Expected an mmCIF .cif file")
                validate_mmcif(path)
            except (ValueError, KeyError, OSError) as error:
                invalid.append({"prediction": list(key), "error": str(error)})
                continue
            accepted.append(dict(row, seed=seed, sample=sample,
                                 ranking_score=ranking, prediction_path=str(path)))
    return accepted, invalid


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def target_key(track, row):
    if track.startswith("monomer_"):
        return (row["pdb_id"].lower(), row["chain_id"])
    return (row["pdb_id"].lower(),
            row.get("interface_chain_id_1", row.get("native_chain_id_1")),
            row.get("interface_chain_id_2", row.get("native_chain_id_2")))


# Interface success is DockQ >= 0.23 with CAPRI's acceptable geometry (Basu & Wallner 2016; Lensink
# et al.): ligand RMSD <= 10 A or interface RMSD <= 4 A. DockQ alone can be passed by collapsing the
# chains onto each other: every native contact is then recovered (fnat ~ 1, DockQ ~ 0.36) while
# iRMSD ~ 15 A and LRMSD ~ 30 A. A row without either RMSD is not a success.
DOCKQ_SUCCESS = 0.23
CAPRI_LRMSD = 10.0
CAPRI_IRMSD = 4.0


def capri_acceptable_geometry(row):
    lrmsd, irmsd = number(row.get("lrmsd")), number(row.get("irmsd"))
    return ((lrmsd is not None and 0 <= lrmsd <= CAPRI_LRMSD)
            or (irmsd is not None and 0 <= irmsd <= CAPRI_IRMSD))


def track_score(track, targets, raw):
    expected = {target_key(track, row) for row in targets}
    if len(expected) != len(targets) or not targets:
        raise ValueError(f"Empty or duplicate target rows: {track}")
    selected = {}
    for row in raw:
        key = target_key(track, row)
        ranking = number(row.get("ranking_score"))
        if key not in expected or ranking is None:
            continue
        if key not in selected or ranking > number(selected[key]["ranking_score"]):
            selected[key] = row
    total, finite_count = 0.0, 0
    for key in expected:
        row = selected.get(key, {})
        if track.startswith("monomer_"):
            value = number(row.get("lddt"))
            valid = value is not None and 0 <= value <= 1
            total += value if valid else 0
        elif track == "interface_protein_ligand":
            rmsd, pli = number(row.get("rmsd")), number(row.get("lddt-pli"))
            valid = rmsd is not None and rmsd >= 0 and pli is not None and 0 <= pli <= 1
            total += int(valid and rmsd < 2.0 and pli > 0.8)
        else:
            value = number(row.get("dockq_score", row.get("dockq")))
            valid = value is not None and 0 <= value <= 1
            total += int(valid and value >= DOCKQ_SUCCESS and capri_acceptable_geometry(row))
        finite_count += int(valid)
    return {"score": total / len(targets), "targets": len(targets),
            "finite_targets": finite_count, "missing_or_invalid": len(targets) - finite_count}


def aggregate_modalities(scores: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Compute the V3 5-modality equal mean (`FOLD_GROUPS`) matching `ssi_index_v3.folding()`."""
    modalities = {}
    for group, names in FOLD_GROUPS.items():
        present = [float(scores[name]) for name in names if name in scores]
        if present:
            modalities[group] = mean(present)
    if not modalities:
        raise ValueError("No scored modalities")
    overall = mean(modalities.values())
    return overall, modalities


def run_inference(command, cwd, log, seconds):
    """Run infer.sh in its own process group; on timeout kill the whole group, workers included."""
    process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        process.wait()
        raise
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def _execute_submission_inference(submission: Path, requests: dict, feature_records: dict, work_output: Path):
    if not (submission / "infer.sh").is_file():
        raise InvalidSubmission("The delivered model must contain infer.sh")
    try:
        audit_submission(submission)
    except (OSError, ValueError) as error:
        raise InvalidSubmission(f"Package audit failed: {error}") from error

    inputs, pred_dir = work_output / "input", work_output / "predictions"
    inputs.mkdir()
    pred_dir.mkdir()
    with (inputs / "requests.jsonl").open("w") as stream:
        for request in requests.values():
            stream.write(json.dumps(request) + "\n")
    write_inference_features(inputs, feature_records)
    setup = submission / "setup.sh"
    try:
        if setup.is_file():
            with (work_output / "setup.log").open("w") as log:
                subprocess.run(["bash", str(setup)], cwd=submission,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        with (work_output / "inference.log").open("w") as log:
            run_inference(["bash", str(submission / "infer.sh"), str(inputs), str(pred_dir)],
                          submission, log, INFERENCE_SECONDS)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise InvalidSubmission(f"Delivered model failed: {error}") from error

    # Reject any symlinks or path traversal in predicted outputs before parsing.
    validate_no_symlinks_or_escapes(pred_dir)
    try:
        accepted, invalid = predictions(pred_dir, requests)
    except (OSError, ValueError, KeyError) as error:
        raise InvalidSubmission(f"Invalid prediction output: {error}") from error
    return accepted, invalid


def _seal_predictions(
    sealed: Path,
    suite_manifest_sha: str,
    accepted: list[dict] | None = None,
    invalid: list[dict] | None = None,
    failure_error: str | None = None,
):
    if sealed.exists():
        raise FileExistsError(f"Sealed directory already exists: {sealed}")
    sealed.parent.mkdir(parents=True, exist_ok=True)
    temporary = sealed.with_name(f".{sealed.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    files = {}
    if failure_error is not None:
        manifest = {
            "schema": SEALED_SCHEMA,
            "status": "invalid_submission",
            "suite_manifest_sha256": suite_manifest_sha,
            "error": failure_error[:4000],
            "files": {},
        }
    else:
        structures_dir = temporary / "structures"
        structures_dir.mkdir()
        relocated_rows = []
        for index, row in enumerate(accepted or []):
            source_path = Path(row["prediction_path"])
            if os.path.islink(source_path) or not source_path.is_file():
                raise InvalidSubmission(f"Refusing to seal non-regular structure: {source_path}")
            rel_dest = Path("structures") / f"{index:06d}.cif"
            dest_path = temporary / rel_dest
            shutil.copyfile(source_path, dest_path, follow_symlinks=False)
            dest_path.chmod(0o444)
            files[rel_dest.as_posix()] = sha256_file(dest_path)
            relocated_rows.append({
                "pdb_id": row["pdb_id"],
                "seed": row["seed"],
                "sample": row["sample"],
                "ranking_score": row["ranking_score"],
                "prediction_path": rel_dest.as_posix(),
            })
        ref_csv = temporary / "prediction_reference.csv"
        with ref_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(relocated_rows)
        ref_csv.chmod(0o444)
        files["prediction_reference.csv"] = sha256_file(ref_csv)
        structures_dir.chmod(0o555)
        manifest = {
            "schema": SEALED_SCHEMA,
            "status": "ok",
            "suite_manifest_sha256": suite_manifest_sha,
            "invalid_predictions": invalid or [],
            "files": files,
        }

    manifest_path = temporary / "sealed_manifest.json"
    write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, sealed)
    ready_tmp = sealed.parent / ".READY.tmp"
    ready_tmp.write_text(sha256_file(sealed / "sealed_manifest.json") + "\n")
    ready_tmp.chmod(0o444)
    os.replace(ready_tmp, sealed.parent / "READY")


def _score_accepted_predictions(
    suite: Path,
    ground_truth: Path,
    targets: dict,
    requests: dict,
    accepted: list[dict],
    invalid: list[dict],
    output: Path,
    *,
    label_isolated_sandbox: bool,
):
    evaluation = output / "evaluation"
    algorithm = evaluation / "submission"
    algorithm.mkdir(parents=True)
    with (algorithm / "prediction_reference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(accepted)

    available = {row["pdb_id"] for row in accepted}
    scoring_targets = output / "scoring_targets"
    scoring_targets.mkdir()
    scored_tracks = []
    for track, rows in targets.items():
        covered = [row for row in rows if row["pdb_id"] in available]
        if covered:
            scored_tracks.append(track)
            with (scoring_targets / f"{track}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(covered)
    if scored_tracks:
        python = Path("/opt/foldbench-env/bin/python")
        environment = dict(os.environ, PATH=f"{python.parent}:{os.environ.get('PATH', '')}")
        with (output / "scoring.log").open("w") as log:
            subprocess.run([
                str(python), "/opt/foldbench/evaluate.py",
                "--targets_dir", str(scoring_targets),
                "--evaluation_dir", str(evaluation), "--algorithm_name", "submission",
                "--ground_truth_dir", str(ground_truth), "--targets", *scored_tracks,
            ], cwd="/opt/foldbench", env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True)

    expected_predictions = sum(len(request["run"]["seeds"]) * request["run"]["num_samples"]
                               for request in requests.values())
    suite_sha = hashlib.sha256((suite / "suite_manifest.json").read_bytes()).hexdigest()
    report = {
        "benchmark_visible_during_development": True,
        "evaluation_mode": "model_replay",
        "label_isolated_sandbox": label_isolated_sandbox,
        "suite_manifest_sha256": suite_sha,
        "expected_predictions": expected_predictions,
        "missing_or_invalid_predictions": expected_predictions - len(accepted),
        "requests": len(requests),
        "accepted_predictions": len(accepted),
        "invalid_predictions": invalid,
        "tracks": {},
    }
    for track, rows in targets.items():
        suffix = "dockqv2" if track in NA_TRACKS else "ost"
        raw = algorithm / "raw" / f"{track}_{suffix}.csv"
        report["tracks"][track] = track_score(track, rows, read_csv(raw) if raw.is_file() else [])
    scores = {track: metrics["score"] for track, metrics in report["tracks"].items()}
    overall_score, modalities = aggregate_modalities(scores)
    report["modalities"] = modalities
    report["score"] = overall_score
    report["reward"] = overall_score
    report["reward_definition"] = (
        "equal-weighted mean across the 5 FoldBench structural modalities (FOLD_GROUPS)"
    )
    write_json(output / "report.json", report)
    write_json(
        output / "reward.json",
        harbor_reward({"reward": overall_score, "score": overall_score, **scores}),
    )
    return report


def run_infer_phase(args):
    if getattr(args, "ground_truth", None) is not None or Path("/benchmark/ground_truths").exists():
        raise RuntimeError("Security violation: ground_truths must NOT be mounted during --phase infer")
    requests = suite_requests(args.suite)
    feature_records = suite_features(args.features, requests)
    suite_targets(args.suite, requests, ground_truth=None)
    suite_sha = hashlib.sha256((args.suite / "suite_manifest.json").read_bytes()).hexdigest()

    try:
        accepted, invalid = _execute_submission_inference(
            args.submission, requests, feature_records, args.output
        )
    except InvalidSubmission as error:
        _seal_predictions(args.sealed, suite_sha, failure_error=str(error))
        if getattr(args, "results", None) is not None:
            _wait_for_grader_results(args.results, args.output, getattr(args, "wait_seconds", 600))
            return
        raise

    _seal_predictions(args.sealed, suite_sha, accepted=accepted, invalid=invalid)
    if getattr(args, "results", None) is not None:
        _wait_for_grader_results(
            args.results, args.output, getattr(args, "wait_seconds", GRADER_WAIT_SECONDS)
        )


def _wait_for_grader_results(results_dir: Path, output_dir: Path, wait_seconds: int):
    deadline = time.monotonic() + wait_seconds
    reward_path = results_dir / "reward.json"
    error_path = results_dir / "error.json"
    while not reward_path.is_file():
        if error_path.is_file() and not reward_path.is_file():
            err_payload = json.loads(error_path.read_text())
            if err_payload.get("type") != "invalid_submission":
                raise RuntimeError(f"Trusted grader failed: {err_payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for trusted grader reward.json")
        time.sleep(0.5)
    for name in ("report.json", "reward.json", "error.json"):
        src = results_dir / name
        if src.is_file():
            shutil.copyfile(src, output_dir / name)


def run_score_phase(args):
    if getattr(args, "submission", None) is not None:
        raise ValueError("Security violation: --submission is forbidden during --phase score")
    wait_seconds = getattr(args, "wait_seconds", GRADER_WAIT_SECONDS)
    ready = args.sealed.parent / "READY"
    deadline = time.monotonic() + wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for sealed predictions READY marker")
        time.sleep(0.5)
    if os.path.islink(ready):
        raise InvalidSubmission("READY marker must not be a symlink")

    # Inspect sealed tree BEFORE reading any content or invoking OpenStructure/DockQ.
    validate_no_symlinks_or_escapes(args.sealed)

    manifest_path = args.sealed / "sealed_manifest.json"
    if not manifest_path.is_file():
        raise InvalidSubmission("Missing sealed_manifest.json")
    if ready.read_text().strip() != sha256_file(manifest_path):
        raise InvalidSubmission("Sealed manifest READY checksum mismatch")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SEALED_SCHEMA:
        raise InvalidSubmission("Invalid sealed manifest schema")

    requests = suite_requests(args.suite)
    targets = suite_targets(args.suite, requests, ground_truth=args.ground_truth)
    suite_sha = hashlib.sha256((args.suite / "suite_manifest.json").read_bytes()).hexdigest()
    if manifest.get("suite_manifest_sha256") != suite_sha:
        raise ValueError("Sealed predictions suite_manifest_sha256 mismatch")

    if manifest.get("status") == "invalid_submission":
        raise InvalidSubmission(manifest.get("error", "Invalid submission sealed by inference phase"))
    if manifest.get("status") != "ok":
        raise InvalidSubmission(f"Unexpected sealed status: {manifest.get('status')}")

    declared_files = manifest.get("files", {})
    if not isinstance(declared_files, dict):
        raise InvalidSubmission("Invalid sealed files mapping")
    actual_files = {
        path.relative_to(args.sealed).as_posix()
        for path in args.sealed.rglob("*")
        if path.is_file() and path.name != "sealed_manifest.json"
    }
    if actual_files != set(declared_files):
        raise InvalidSubmission("Undeclared or missing files in sealed directory")
    for rel_path, expected_sha in declared_files.items():
        file_path = contained_file(args.sealed, rel_path)
        if sha256_file(file_path) != expected_sha:
            raise InvalidSubmission(f"Sealed file checksum mismatch: {rel_path}")

    try:
        accepted, extra_invalid = predictions(args.sealed, requests)
    except (OSError, ValueError, KeyError) as error:
        raise InvalidSubmission(f"Sealed predictions failed validation: {error}") from error

    invalid = list(manifest.get("invalid_predictions", [])) + extra_invalid
    return _score_accepted_predictions(
        args.suite,
        args.ground_truth,
        targets,
        requests,
        accepted,
        invalid,
        args.output,
        label_isolated_sandbox=True,
    )


def run(args):
    phase = getattr(args, "phase", "all")
    if phase == "infer":
        return run_infer_phase(args)
    if phase == "score":
        return run_score_phase(args)
    if phase != "all":
        raise ValueError(f"Unsupported phase: {phase}")

    # Legacy single-process dev self-test (`--phase all`, label_isolated_sandbox=False).
    requests = suite_requests(args.suite)
    feature_records = suite_features(args.features, requests)
    targets = suite_targets(args.suite, requests, ground_truth=args.ground_truth)
    accepted, invalid = _execute_submission_inference(
        args.submission, requests, feature_records, args.output
    )
    return _score_accepted_predictions(
        args.suite,
        args.ground_truth,
        targets,
        requests,
        accepted,
        invalid,
        args.output,
        label_isolated_sandbox=False,
    )


def write_invalid_submission(output: Path, error: Exception, *, label_isolated_sandbox: bool):
    write_json(output / "error.json", {"error": str(error), "type": "invalid_submission"})
    write_json(output / "report.json", {
        "status": "invalid_submission",
        "error": str(error)[:4000],
        "reward": 0.0,
        "score": 0.0,
        "label_isolated_sandbox": label_isolated_sandbox,
    })
    write_json(output / "reward.json", {"reward": 0.0, "score": 0.0, "invalid_submission": 1.0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "score", "all"), default="all")
    for flag in ("submission", "suite", "features", "ground-truth", "sealed", "results", "output"):
        parser.add_argument(
            "--" + flag,
            type=lambda value: Path(value).resolve(),
            required=flag in {"suite", "output"},
        )
    parser.add_argument("--wait-seconds", type=int, default=GRADER_WAIT_SECONDS)
    args = parser.parse_args()

    if args.phase == "infer":
        for req in ("submission", "features", "sealed"):
            if getattr(args, req) is None:
                parser.error(f"--{req} is required when --phase infer")
        if args.ground_truth is not None:
            parser.error("--ground-truth must not be passed when --phase infer")
    elif args.phase == "score":
        for req in ("ground_truth", "sealed"):
            if getattr(args, req.replace("-", "_")) is None:
                parser.error(f"--{req.replace('_', '-')} is required when --phase score")
        if args.submission is not None:
            parser.error("--submission is forbidden when --phase score")
    else:
        for req in ("submission", "features", "ground_truth"):
            if getattr(args, req) is None:
                parser.error(f"--{req.replace('_', '-')} is required when --phase all")

    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("reward.json", "report.json", "input", "predictions", "evaluation")):
        parser.error("Use a fresh verifier output directory; refusing to mix runs")
    try:
        run(args)
    except InvalidSubmission as error:
        write_invalid_submission(
            args.output, error, label_isolated_sandbox=(args.phase != "all")
        )
        print(f"invalid submission: {error}")
    except Exception as error:
        write_json(args.output / "error.json", {"error": str(error), "type": type(error).__name__})
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
