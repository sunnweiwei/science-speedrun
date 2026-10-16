"""Deterministic chunking and strict result assembly for independent GPU workers."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from common import PROTOCOL, digest, indexed, read_rows, write_json, write_rows

CHUNK_SIZES = {"discovery": 128, "phonons": 4, "elasticity": 8}


def chunk_plan(cases):
    indexed(cases)
    groups = defaultdict(list)
    for case in cases:
        if case["suite"] not in CHUNK_SIZES:
            raise ValueError(f"Unknown suite: {case['suite']}")
        groups[case["suite"]].append(case)
    per_suite = {}
    for suite, rows in groups.items():
        rows.sort(
            key=lambda r: hashlib.sha256(
                f"gpu-chunks-v1:{r['id']}".encode()
            ).hexdigest()
        )
        size = CHUNK_SIZES[suite]
        per_suite[suite] = [rows[i : i + size] for i in range(0, len(rows), size)]
    # Interleave suites; the cloud queue sends each next chunk to any free GPU.
    chunks = []
    for i in range(max(map(len, per_suite.values()))):
        for suite in CHUNK_SIZES:
            if i >= len(per_suite.get(suite, [])):
                continue
            rows = per_suite[suite][i]
            payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
            sha = hashlib.sha256(payload.encode()).hexdigest()
            chunks.append(
                {
                    "chunk_id": f"{suite}-{i:05d}-{sha[:12]}",
                    "suite": suite,
                    "input_sha256": sha,
                    "case_ids": [r["id"] for r in rows],
                }
            )
    flattened = [key for chunk in chunks for key in chunk["case_ids"]]
    if len(flattened) != len(cases) or set(flattened) != {r["id"] for r in cases}:
        raise ValueError("Chunk plan lost or duplicated cases")
    return chunks


def merge(inputs: Path, chunks_root: Path, plan, output: Path, checkpoint_sha256: str,
          unfinished_reason: str | None = None):
    """Merge finished chunks. With `unfinished_reason` (the verifier's time limit), a chunk that never
    finished contributes one failed record per case instead of stopping the merge."""
    if output.exists():
        raise FileExistsError(output)
    cases = indexed(read_rows(inputs))
    expected_ids = [key for chunk in plan for key in chunk["case_ids"]]
    if len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != cases.keys():
        raise ValueError("Chunk plan does not exactly cover evaluation IDs")
    all_predictions, chunk_reports = [], []
    reference_metadata = None
    required_same = (
        "factory",
        "checkpoint_sha256",
        "versions",
        "energy_audit",
        "energy_audit_sha256",
    )
    unfinished = 0
    for chunk in plan:
        directory = chunks_root / chunk["chunk_id"]
        if unfinished_reason and not (directory / "worker.json").is_file():
            all_predictions.extend({"id": key, "suite": cases[key]["suite"], "status": "failed",
                                    "error": unfinished_reason, "seconds": 0.0} for key in chunk["case_ids"])
            unfinished += len(chunk["case_ids"])
            continue
        metadata = json.loads((directory / "run.json").read_text())
        worker = json.loads((directory / "worker.json").read_text())
        if metadata["protocol"] != PROTOCOL or not metadata["complete"]:
            raise ValueError(
                f"Incomplete/protocol-mismatched chunk: {chunk['chunk_id']}"
            )
        if metadata["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Mixed model checkpoints")
        if worker["chunk_input_sha256"] != chunk["input_sha256"]:
            raise ValueError("Wrong chunk input identity")
        if digest(directory / "predictions.jsonl") != metadata["predictions_sha256"]:
            raise ValueError("Corrupt chunk predictions")
        if digest(directory / "inputs.jsonl") != metadata["inputs_sha256"]:
            raise ValueError("Corrupt chunk inputs")
        chunk_inputs = read_rows(directory / "inputs.jsonl")
        if chunk_inputs != [cases[key] for key in chunk["case_ids"]]:
            raise ValueError("Chunk geometry/order differs from canonical inputs")
        predictions = read_rows(directory / "predictions.jsonl")
        ids = indexed(predictions)
        if set(ids) != set(chunk["case_ids"]) or metadata["cases"] != len(ids):
            raise ValueError("Missing/extra cases in chunk output")
        if any(r["suite"] != cases[r["id"]]["suite"] for r in predictions):
            raise ValueError("Prediction suite mismatch")
        comparable = {key: metadata.get(key) for key in required_same}
        if reference_metadata is None:
            reference_metadata = comparable
        elif reference_metadata != comparable:
            raise ValueError(
                "Workers used inconsistent environments or energy conventions"
            )
        all_predictions.extend(predictions)
        chunk_reports.append(worker)
    indexed(all_predictions)
    all_predictions.sort(key=lambda r: r["id"])
    write_rows(output / "predictions.jsonl", all_predictions)
    metadata = {
        "protocol": PROTOCOL,
        **(reference_metadata or {}),
        "inputs_sha256": digest(inputs),
        "cases": len(all_predictions),
        "complete": True,
        "device": "cuda",
        "distributed": True,
        "predictions_sha256": digest(output / "predictions.jsonl"),
        "unfinished_cases": unfinished,
    }
    write_json(output / "run.json", metadata)
    workers = {r["worker_id"] for r in chunk_reports}
    timing = {
        "worker_instances": len(workers),
        "chunks": len(chunk_reports),
        "suites": {},
    }
    for suite in CHUNK_SIZES:
        rows = [r for r in all_predictions if r["suite"] == suite]
        if not rows:
            continue
        durations = sorted(r["seconds"] for r in rows)
        timing["suites"][suite] = {
            "cases": len(rows),
            "successful": sum(r["status"] == "ok" for r in rows),
            "failed": sum(r["status"] != "ok" for r in rows),
            "total_case_seconds": sum(durations),
            "mean_case_seconds": sum(durations) / len(rows),
            "median_case_seconds": durations[len(rows) // 2],
            "max_case_seconds": max(durations),
            "nonconverged_initial": sum(
                r["status"] == "ok"
                and not r.get("diagnostics", {}).get("converged", False)
                for r in rows
            ),
        }
    timing["coverage"] = dict(Counter(r["status"] for r in all_predictions))
    timing["case_gpu_worker_hours"] = sum(r["seconds"] for r in all_predictions) / 3600
    timing["note"] = (
        "Case time includes CPU portions while a GPU worker is reserved; this is not provider-billed time. Failed cases are retained."
    )
    write_json(output / "timing.json", timing)
    return timing
