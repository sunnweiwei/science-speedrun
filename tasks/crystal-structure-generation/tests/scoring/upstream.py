"""Run and normalize the pinned LeMat-GenBench aggregate scorer."""

import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def final_scores(value: str) -> dict:
    marker = "final_scores="
    start = value.find(marker)
    if start < 0:
        return {}
    start = value.find("{", start + len(marker))
    if start < 0:
        return {}
    depth = 0
    quote = None
    escape = False
    end = None
    for index in range(start, len(value)):
        char = value[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return {}
    text = value[start:end]
    text = re.sub(r"np\.(?:float64|float32|int64|int32)\(([^)]+)\)", r"\1", text)
    text = re.sub(r"\bnan\b", "None", text)
    parsed = ast.literal_eval(text)
    return parsed if isinstance(parsed, dict) else {}


def normalize_result(document: dict, expected: int) -> tuple[dict, dict]:
    results = document.get("results", {})
    validity = document.get("validity_filtering", {})
    valid_count = int(validity.get("valid_structures", 0) or 0)
    if valid_count:
        # Upstream sometimes serializes infrastructure failures inside a
        # successful BenchmarkResult. Never turn an unavailable scorer into 0.
        for value in results.values():
            if isinstance(value, str) and "No embeddings found for model " in value:
                raise RuntimeError("LeMat required MLIP output is missing")
        aliases = {
            "sun": ("sun", "sun_new"),
            "novelty": ("novelty", "novelty_new"),
            "uniqueness": ("uniqueness", "uniqueness_new"),
        }
        for family, names in aliases.items():
            matches = [results[name] for name in names if name in results]
            if not matches or any(isinstance(value, dict) and "error" in value for value in matches):
                raise RuntimeError(f"LeMat {family} benchmark failed or produced no result")
    sun = final_scores(results.get("sun_new", results.get("sun", "")))
    novelty = final_scores(results.get("novelty", results.get("novelty_new", "")))
    uniqueness = final_scores(results.get("uniqueness", results.get("uniqueness_new", "")))
    if valid_count:
        for family, scores in (("sun", sun), ("novelty", novelty), ("uniqueness", uniqueness)):
            for key, value in scores.items():
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    raise RuntimeError(f"LeMat {family} returned nonfinite {key}")
            evaluated = scores.get("total_structures_evaluated")
            if evaluated is not None and int(evaluated) != valid_count:
                raise RuntimeError(f"LeMat {family} did not evaluate every valid structure")
        for scores, names in ((sun, ("sun_count", "msun_count", "stable_count", "metastable_count")),
                              (novelty, ("novel_structures_count",)),
                              (uniqueness, ("unique_structures_count",))):
            for key in names:
                if key in scores and (not isinstance(scores[key], (int, float))
                                      or not 0 <= scores[key] <= valid_count
                                      or int(scores[key]) != scores[key]):
                    raise RuntimeError(f"LeMat returned invalid {key}")
    if valid_count and (
        not {"sun_count", "msun_count"}.issubset(sun)
        or "novel_structures_count" not in novelty
        or "unique_structures_count" not in uniqueness
    ):
        raise RuntimeError("LeMat result could not be parsed into required score counts")
    counts = {
        "expected": expected,
        "loaded": int(document.get("run_info", {}).get("n_structures", 0) or 0),
        "valid": valid_count,
        "stable": int(sun.get("stable_count", 0) or 0),
        "metastable": int(sun.get("metastable_count", 0) or 0),
        "sun": int(sun.get("sun_count", 0) or 0),
        "metasun": int(sun.get("msun_count", 0) or 0),
        "novel": int(novelty.get("novel_structures_count", 0) or 0),
        "unique": int(uniqueness.get("unique_structures_count", 0) or 0),
    }
    rates = {
        "validity": counts["valid"] / expected if expected else 0.0,
        "stable": counts["stable"] / expected if expected else 0.0,
        "metastable": counts["metastable"] / expected if expected else 0.0,
        "sun": counts["sun"] / expected if expected else 0.0,
        "metasun": counts["metasun"] / expected if expected else 0.0,
        "sun_metasun": (counts["sun"] + counts["metasun"]) / expected if expected else 0.0,
        "novelty": counts["novel"] / expected if expected else 0.0,
        "uniqueness": counts["unique"] / expected if expected else 0.0,
    }
    return rates, {"counts": counts, "upstream": document}


def stage_records(records: dict[tuple[str, int], Path], request_ids: set[str], destination: Path) -> int:
    from pymatgen.core import Structure

    destination.mkdir(parents=True)
    count = 0
    for (request_id, sample_index), source in sorted(records.items()):
        if request_id not in request_ids:
            continue
        try:
            Structure.from_file(source)
        except Exception:
            continue
        shutil.copyfile(source, destination / f"{request_id}--{sample_index:06d}.cif")
        count += 1
    return count


def writable_datasets_cache(work_root: Path) -> Path | None:
    """Give `datasets` a cache directory it can lock while the assets stay read-only.

    The pinned LeMat scorer loads the hull tables and the LeMat-Bulk novelty
    reference through `datasets.load_dataset`, which creates lock files inside
    HF_DATASETS_CACHE even for a fully cached, offline load. The scorer assets
    are mounted read-only in both the development and the verifier containers,
    so that lock raises EROFS; LeMat swallows the error, falls back to a Hub
    path that does not exist, and every e_above_hull becomes None, which turns
    SUN/MetaSUN into a silent zero. Return a writable directory containing a
    symlink per cached dataset, or None when the configured cache is writable.
    """
    source = Path(os.environ.get("HF_DATASETS_CACHE", "")) if os.environ.get("HF_DATASETS_CACHE") else None
    if source is None or not source.is_dir() or os.access(source, os.W_OK):
        return None
    cache = work_root / "datasets-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.is_dir() and not (cache / entry.name).exists():
            (cache / entry.name).symlink_to(entry.resolve())
    return cache


def run_lemat(
    records: dict[tuple[str, int], Path],
    request_ids: set[str],
    expected: int,
    work_root: Path,
    scorer_root: Path,
    python: Path,
) -> tuple[dict, dict]:
    cif_root = work_root / "cifs"
    produced = stage_records(records, request_ids, cif_root)
    if produced == 0:
        names = (
            "validity", "stable", "metastable", "sun", "metasun",
            "sun_metasun", "novelty", "uniqueness",
        )
        empty = {name: 0.0 for name in names}
        counts = {
            "expected": expected, "loaded": 0, "valid": 0, "stable": 0,
            "metastable": 0, "sun": 0, "metasun": 0, "novel": 0, "unique": 0,
        }
        return empty, {"counts": counts}
    result_root = scorer_root / "results_final"
    before = set(result_root.glob("crystalgen-*.json")) if result_root.exists() else set()
    name = f"crystalgen-{os.getpid()}-{time.time_ns()}"
    command = [
        str(python), str(scorer_root / "scripts/run_benchmarks.py"),
        "--cifs", str(cif_root), "--config", "comprehensive_multi_mlip_hull", "--name", name,
    ]
    env = dict(os.environ)
    cache = writable_datasets_cache(work_root)
    if cache is not None:
        env["HF_DATASETS_CACHE"] = str(cache)
    subprocess.run(command, cwd=scorer_root, check=True, env=env)
    created = sorted(set(result_root.glob(f"{name}_*.json")) - before, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1:
        raise RuntimeError(f"expected one LeMat result, found {len(created)}")
    document = json.loads(created[0].read_text())
    return normalize_result(document, expected)
