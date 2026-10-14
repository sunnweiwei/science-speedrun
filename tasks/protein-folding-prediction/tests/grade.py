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
import struct
import subprocess
import sys
import traceback


TRACKS = (
    "monomer_protein", "monomer_rna", "monomer_dna",
    "interface_protein_protein", "interface_antibody_antigen",
    "interface_protein_peptide", "interface_protein_rna",
    "interface_protein_dna", "interface_protein_ligand",
)
NA_TRACKS = {"interface_protein_rna", "interface_protein_dna"}
FIELDS = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"]
FEATURE_VERSION = "esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1"
FEATURE_WIDTH = 2560
FEATURE_GROUP_SIZE = 64
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
# environment/contract.md "Resources": the whole batch must finish within this wall time.
INFERENCE_SECONDS = 8 * 3600
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


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
    with (output / "prediction_reference.csv").open(newline="") as stream:
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


def run(args):
    requests = suite_requests(args.suite)
    feature_records = suite_features(args.features, requests)
    targets = {track: read_csv(args.suite / "targets" / f"{track}.csv")
               for track in TRACKS if (args.suite / "targets" / f"{track}.csv").is_file()}
    targets = {track: rows for track, rows in targets.items() if rows}
    if not targets:
        raise ValueError("No scoring targets")
    for rows in targets.values():
        for row in rows:
            if row["pdb_id"] not in requests:
                raise ValueError("Target table references an unknown request")
            if not (args.ground_truth / (row["pdb_id"] + ".cif")).is_file():
                raise FileNotFoundError(f"Missing native structure: {row['pdb_id']}")
    if not (args.submission / "infer.sh").is_file():
        raise InvalidSubmission("The delivered model must contain infer.sh")
    try:
        audit_submission(args.submission)
    except (OSError, ValueError) as error:
        raise InvalidSubmission(f"Package audit failed: {error}") from error

    # Separate directories prevent stale prior predictions from being scored.
    inputs, output = args.output / "input", args.output / "predictions"
    inputs.mkdir()
    output.mkdir()
    with (inputs / "requests.jsonl").open("w") as stream:
        for request in requests.values():
            stream.write(json.dumps(request) + "\n")
    write_inference_features(inputs, feature_records)
    setup = args.submission / "setup.sh"
    try:
        if setup.is_file():
            with (args.output / "setup.log").open("w") as log:
                subprocess.run(["bash", str(setup)], cwd=args.submission,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        with (args.output / "inference.log").open("w") as log:
            run_inference(["bash", str(args.submission / "infer.sh"), str(inputs), str(output)],
                          args.submission, log, INFERENCE_SECONDS)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise InvalidSubmission(f"Delivered model failed: {error}") from error
    accepted, invalid = predictions(output, requests)
    evaluation = args.output / "evaluation"
    algorithm = evaluation / "submission"
    algorithm.mkdir(parents=True)
    with (algorithm / "prediction_reference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(accepted)

    available = {row["pdb_id"] for row in accepted}
    scoring_targets = args.output / "scoring_targets"
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
        with (args.output / "scoring.log").open("w") as log:
            subprocess.run([
                str(python), "/opt/foldbench/evaluate.py",
                "--targets_dir", str(scoring_targets),
                "--evaluation_dir", str(evaluation), "--algorithm_name", "submission",
                "--ground_truth_dir", str(args.ground_truth), "--targets", *scored_tracks,
            ], cwd="/opt/foldbench", env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True)

    expected_predictions = sum(len(request["run"]["seeds"]) * request["run"]["num_samples"]
                               for request in requests.values())
    report = {"benchmark_visible_during_development": True,
              "evaluation_mode": "model_replay", "label_isolated_sandbox": False,
              "suite_manifest_sha256": hashlib.sha256((args.suite / "suite_manifest.json").read_bytes()).hexdigest(),
              "expected_predictions": expected_predictions,
              "missing_or_invalid_predictions": expected_predictions - len(accepted),
              "requests": len(requests), "accepted_predictions": len(accepted),
              "invalid_predictions": invalid, "tracks": {}}
    for track, rows in targets.items():
        suffix = "dockqv2" if track in NA_TRACKS else "ost"
        raw = algorithm / "raw" / f"{track}_{suffix}.csv"
        report["tracks"][track] = track_score(track, rows, read_csv(raw) if raw.is_file() else [])
    scores = {track: metrics["score"] for track, metrics in report["tracks"].items()}
    # reward (TASK_STANDARD.md): the unweighted mean of the scored tracks' scores, each already in [0, 1]
    report["reward"] = sum(scores.values()) / len(scores)
    report["reward_definition"] = "unweighted mean of the per-track scores in reward.json"
    write_json(args.output / "report.json", report)
    write_json(args.output / "reward.json", harbor_reward({"reward": report["reward"], **scores}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("submission", "suite", "features", "ground-truth", "output"):
        parser.add_argument("--" + flag, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("reward.json", "report.json", "input", "predictions", "evaluation")):
        parser.error("Use a fresh verifier output directory; refusing to mix runs")
    try:
        run(args)
    except InvalidSubmission as error:
        # An invalid package or a failing setup.sh/infer.sh scores 0 (TASK_STANDARD.md); Harbor needs reward.json.
        write_json(args.output / "error.json", {"error": str(error), "type": "invalid_submission"})
        write_json(args.output / "report.json", {"status": "invalid_submission", "error": str(error)[:4000], "reward": 0.0})
        write_json(args.output / "reward.json", {"reward": 0.0, "invalid_submission": 1.0})
        print(f"invalid submission: {error}")
    except Exception as error:
        write_json(args.output / "error.json", {"error": str(error), "type": type(error).__name__})
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
