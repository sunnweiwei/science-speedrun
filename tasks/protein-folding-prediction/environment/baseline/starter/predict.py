"""Optional, framework-free batch adapter for the task's inference contract."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import traceback

# Large assemblies follow hundreds of small ones in one replay; without
# expandable segments the caching allocator's fragmentation alone pushes a
# 2,500-token request over 80 GB. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from model import load_model  # noqa: E402


def feature_paths(input_dir):
    records = {}
    with (input_dir / "features.esmc6b_final_int8.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            digest = record["sequence_sha256"]
            if digest in records:
                raise ValueError(f"Duplicate feature digest: {digest}")
            path = (input_dir / record["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            records[digest] = path
    return records


def record_failure(path, request_id, seed, sample, error):
    """Append one line per failed prediction; the scorer counts it as missing."""
    with path.open("a") as destination:
        destination.write(json.dumps({
            "request_id": request_id,
            "seed": seed,
            "sample": sample,
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0][:500] if str(error) else "",
        }) + "\n")
    print(f"prediction failed for {request_id} seed={seed} sample={sample}: "
          f"{type(error).__name__}", flush=True)
    traceback.print_exc()


def release_accelerator_memory():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in the task image
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    structures = args.output / "structures"
    structures.mkdir(exist_ok=True)
    model = load_model(Path(__file__).resolve().parent)
    features = feature_paths(args.input)
    seen = set()
    failures = args.output / "failures.jsonl"
    with (args.input / "requests.jsonl").open() as source, (
        args.output / "prediction_reference.csv"
    ).open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=[
            "pdb_id", "seed", "sample", "ranking_score", "prediction_path"
        ])
        writer.writeheader()
        for line in source:
            request = json.loads(line)
            request_id = request["request_id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id) or request_id in seen:
                raise ValueError("Unsafe or duplicate request ID")
            seen.add(request_id)
            requested_features = {}
            for entity in request["target"]["entities"]:
                if entity["type"] != "protein":
                    continue
                digest = hashlib.sha256(entity["sequence"].encode("utf-8")).hexdigest()
                requested_features[digest] = features[digest]
            for seed in request["run"]["seeds"]:
                for sample in range(request["run"]["num_samples"]):
                    try:
                        result = model.predict(request=request, feature_paths=requested_features,
                                               seed=seed, sample_index=sample)
                        score = float(result.get("ranking_score", 0.0))
                        if not math.isfinite(score):
                            raise ValueError("ranking_score must be finite")
                        relative = Path("structures") / f"{request_id}-{seed}-{sample}.cif"
                        (args.output / relative).write_text(result["mmcif"])
                    except Exception as error:  # noqa: BLE001 - one target must not abort the batch
                        # The contract keeps every target in the scorer's denominator, so a
                        # request the model cannot serve (typically CUDA out of memory on a
                        # large assembly) is recorded as a missing prediction rather than
                        # turning the whole batch into an infrastructure failure.
                        record_failure(failures, request_id, seed, sample, error)
                        release_accelerator_memory()
                        continue
                    writer.writerow(dict(pdb_id=request_id, seed=seed, sample=sample,
                                         ranking_score=score, prediction_path=relative.as_posix()))
                    destination.flush()


if __name__ == "__main__":
    main()
