"""Run and normalize the pinned LeMat-GenBench aggregate scorer."""

import ast
import hashlib
from . import displacement
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


SCORER_FILES = {
    "scripts/run_benchmarks.py": "754e8b22230c47346c24d4f457eef3bffff112e4170326f2f5fff5be8b5a656d",
    "src/lemat_genbench/models/mace/calculator.py": "8a07caee3625f1aa47d64adf772428e35c7c62673ea9a14434a4050e97cafccf",
    "src/lemat_genbench/preprocess/multi_mlip_preprocess.py": "1c0231dadbe94e67528fc23bfbb87290e581ca703bf1a32dc43dad9f333999f2",
    "src/lemat_genbench/fingerprinting/reference_index.py": "408720c42b1b531c9818e1e9acf66545ab03828fd82113b7ef98684bb8d424ec",
    "src/lemat_genbench/fingerprinting/substitution.py": "5ea2db165c9da140876e4c1974bbbaac3b205bde17bc4fd271e3e4dfee05c7e1",
    "src/lemat_genbench/metrics/novelty_metric.py": "f63e497b563b8363c364bf708d7816193c17e68aeb4cbbb4a9ca17db0aa8eb86",
    "src/lemat_genbench/metrics/uniqueness_metric.py": "ccde9ed37cab00e0a12304be8024388facb288994513e650873dc080e1d2ce2c",
    "src/lemat_genbench/metrics/sun_metric.py": "9bd83e1e12370acd12968f6e6bb13b7d76a2fb7b6844d4b3421d05fa72f5c8a1"
}


def verify_scorer(root):
    """Reject historical matcher and energy implementations even if the reported metric keys coincide."""
    for relative, expected in SCORER_FILES.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected submitted-geometry scorer implementation: {relative}")


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
    # A missing count is a scorer failure, not "zero valid structures".
    if (not isinstance(document.get("run_info"), dict) or "n_structures" not in document["run_info"]
            or not isinstance(validity, dict) or "valid_structures" not in validity):
        raise RuntimeError("LeMat result lacks structure counts; refusing to score it as zero")
    loaded, valid_count = document["run_info"]["n_structures"], validity["valid_structures"]
    if (type(loaded) is not int or type(valid_count) is not int
            or not 0 <= valid_count <= loaded <= expected):
        raise RuntimeError("LeMat returned invalid loaded or valid structure counts")
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
    if not valid_count and any(counts[key] for key in ("stable", "metastable", "sun", "metasun", "novel", "unique")):
        raise RuntimeError("LeMat reports accepted structures but zero valid structures")
    strict_sun = counts["sun"] / expected if expected else 0.0
    sun_plus_metasun = (counts["sun"] + counts["metasun"]) / expected if expected else 0.0
    rates = {
        "validity": counts["valid"] / expected if expected else 0.0,
        "stable": counts["stable"] / expected if expected else 0.0,
        "metastable": counts["metastable"] / expected if expected else 0.0,
        "sun": strict_sun,
        "metasun": counts["metasun"] / expected if expected else 0.0,
        "sun_metasun": sun_plus_metasun,
        "ssi_score": 0.5 * (strict_sun + sun_plus_metasun),
        "novelty": counts["novel"] / expected if expected else 0.0,
        "uniqueness": counts["unique"] / expected if expected else 0.0,
    }
    details = {"counts": counts, "upstream": document}
    # The published SUN and MSUN bins each deduplicate their own members.
    # Upstream also computes one-pool deduplication across all low-energy structures.
    global_count = sun.get("metasun_count")
    if global_count is not None:
        if type(global_count) is not int or not 0 <= global_count <= valid_count:
            raise RuntimeError("Invalid whole-pool low-energy uniqueness count")
        details["global_unique_novel_low_energy_count"] = global_count
        rates["global_unique_novel_low_energy"] = global_count / expected if expected else 0.0
    return rates, details


def check_hull_ensemble(document, valid_count):
    """Every evaluated input geometry must have a finite, three-model hull mean."""
    if not valid_count:
        return
    from .experimental_v2 import _literal

    result = document.get("results", {}).get("stability")
    if not isinstance(result, str):
        raise RuntimeError("LeMat did not return its raw geometry energy report")
    try:
        values = _literal(result)["evaluator_results"]["mean_e_above_hull"]["metric_results"]["mean_e_above_hull"]["individual_values"]
    except (KeyError, ValueError, TypeError) as error:
        raise RuntimeError("LeMat did not return per-structure hull energies") from error
    if len(values) != valid_count:
        raise RuntimeError("Raw-geometry hull energies do not cover every valid structure")
    for i, row in enumerate(values):
        energies = [row.get(f"value_{model}") for model in ("orb", "mace", "uma")]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in energies):
            raise RuntimeError(f"Missing or nonfinite MLIP hull energy at valid slot {i}")
        reported = row.get("value")
        if not isinstance(reported, (int, float)) or not math.isclose(sum(energies)/3, reported, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError(f"Invalid three-model hull mean at valid slot {i}")


def stage_records(records: dict[tuple[str, int], Path], request_ids: set[str], destination: Path) -> int:
    from pymatgen.core import Structure

    destination.mkdir(parents=True)
    count = 0
    for (request_id, sample_index), source in sorted(records.items()):
        if request_id not in request_ids:
            continue
        try:
            Structure.from_file(source)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError, SyntaxError):
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
    rmsd_slots=(),
) -> tuple[dict, dict]:
    verify_scorer(scorer_root)
    cif_root = work_root / "cifs"
    produced = stage_records(records, request_ids, cif_root)
    if produced == 0:
        names = (
            "validity", "stable", "metastable", "sun", "metasun",
            "sun_metasun", "ssi_score", "novelty", "uniqueness",
        )
        empty = {name: 0.0 for name in names}
        counts = {
            "expected": expected, "loaded": 0, "valid": 0, "stable": 0,
            "metastable": 0, "sun": 0, "metasun": 0, "novel": 0, "unique": 0,
        }
        return empty, {"counts": counts, "displacement": displacement.summarize(rmsd_slots, {})}
    result_root = scorer_root / "results_final"
    before = set(result_root.glob("crystalgen-*.json")) if result_root.exists() else set()
    name = f"crystalgen-{os.getpid()}-{time.time_ns()}"
    # Preserve sealed request/sample order across filesystems and machines.
    cif_list = work_root / "cif-paths.txt"
    cif_list.write_text("".join(str(p.resolve()) + "\n" for p in sorted(cif_root.iterdir())))
    selection = work_root / "rmsd-selection.json"
    selection.write_text(json.dumps(list(rmsd_slots)))
    raw_rmsd = work_root / "rmsd-observations.json"
    command = [
        str(python), str(scorer_root / "scripts/run_benchmarks.py"),
        "--cifs", str(cif_list), "--config", "comprehensive_multi_mlip_hull", "--name", name,
    ]
    env = dict(os.environ, CRYSTALGEN_RMSD_SELECTION=str(selection.resolve()),
               CRYSTALGEN_RMSD_OUTPUT=str(raw_rmsd.resolve()))
    cache = writable_datasets_cache(work_root)
    if cache is not None:
        env["HF_DATASETS_CACHE"] = str(cache)
    subprocess.run(command, cwd=scorer_root, check=True, env=env)
    created = sorted(set(result_root.glob(f"{name}_*.json")) - before, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1:
        raise RuntimeError(f"expected one LeMat result, found {len(created)}")
    document = json.loads(created[0].read_text())
    rates, details = normalize_result(document, expected)
    if details["counts"]["loaded"] != produced:
        raise RuntimeError("LeMat did not load every CIF that passed initial parsing")
    check_hull_ensemble(document, details["counts"]["valid"])
    if details["counts"]["valid"] and not raw_rmsd.is_file():
        raise RuntimeError("Patched LeMat runner did not write sampled-displacement receipts")
    measurements = json.loads(raw_rmsd.read_text()) if raw_rmsd.is_file() else {}
    details["displacement"] = displacement.summarize(rmsd_slots, measurements)
    return rates, details
