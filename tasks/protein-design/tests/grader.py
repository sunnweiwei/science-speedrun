#!/usr/bin/env python3
"""Trusted designability grader.

Consumes sealed backbone coordinates and the public conditioning contexts. It never
imports or executes the submission.

Failure policy
--------------
An individual design that is malformed or geometrically invalid counts as a FAILED
design, not a void track: for a generative model, per-sample failure is the
measurement. A malformed *index* (bad schema, unknown or duplicate example_id) is a
contract violation and zeroes that track. Designs the model never emitted stay in the
denominator, so an empty submission scores zero rather than scoring nothing.

A scorer that cannot run at all is an infrastructure error and exits nonzero; it never
degrades into a zero pass rate.

Reporting
---------
Every track reports a `pass_rate` - the mean per-target fraction of requested designs
that satisfy the track's PUBLISHED criterion, and nothing besides it - with `diversity`
alongside. The task applied geometry and conditioning gates of its own before folding until
2026-09-26; they are gone, because they discarded most of the signal while inventing a
second criterion nobody publishes. See environment/evaluation/RESULTS.md. There is deliberately NO
composite: the four tracks are judged by different published criteria, and the
`unconditional` track is synthetic to this task, so any weighted mean would be a number
this task invented rather than one it measured. See environment/evaluation/RESULTS.md.
"""

import argparse
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from common import (DESIGN_FIELDS, LIGAND_CLASH_MIN, MOTIF_RMSD_MAX, MOTIF_TRACKS, NOVELTY_MAX_TM,
                    PLDDT_MIN, REPORT_SCHEMA, SC_RMSD_MAX, SNAPSHOT_SCHEMA, TRACKS,
                    jsonl, sha256, write_json)
from geometry import (cluster_designs, minimum_distance, parse_allatom_pdb)
from pipeline import ScorerUnavailable

MAX_CONTEXT_BYTES = 64 * 1024 * 1024


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def load_context(path):
    """Load one conditioning context; pickle is disabled and shapes are checked."""
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_CONTEXT_BYTES:
        raise ValueError(f"unsafe context file: {path.name}")
    with np.load(path, allow_pickle=False) as handle:
        context = {key: handle[key] for key in handle.files}
    for key, value in context.items():
        # Residue-name arrays are text; only numeric arrays get a finiteness check,
        # and only numeric arrays are promoted to float64 for the geometry code.
        if value.dtype.kind in "fiu":
            context[key] = value.astype(np.float64)
            if not np.all(np.isfinite(context[key])):
                raise ValueError(f"non-finite context array: {key}")
        elif value.dtype.kind not in "USb":     # text labels and boolean masks
            raise ValueError(f"unsupported context array type: {key}")
    return context


def validate_request(track, structure, motif_map, context):
    """Did the submission answer the request? Returns (reason_or_None, detail dict).

    This is contract validation, NOT a scoring criterion. The motif map has to name one
    distinct in-range residue per motif position, carrying the requested residue
    identities, or `pipeline._motif_recapitulation` has nothing to measure against.
    Nothing here judges quality: geometry and placement are settled by the refold.
    """
    detail = {}
    if track not in MOTIF_TRACKS:
        return None, detail
    coordinates, _, restypes = structure
    motif = context.get("motif_coords")
    motif_types = context.get("motif_restypes")
    if motif is None or context.get("motif_mask") is None or motif_types is None:
        raise ValueError("context is missing the motif specification")
    size = motif.shape[0]
    if (not isinstance(motif_map, list) or len(motif_map) != size
            or any(type(index) is not int for index in motif_map)
            or len(set(motif_map)) != size
            or any(not 0 <= index < coordinates.shape[0] for index in motif_map)):
        return "motif_map must list one distinct in-range residue per motif position", detail
    requested = [str(name) for name in motif_types]
    placed = [restypes[index] for index in motif_map]
    if placed != requested:
        return (f"motif residue identities differ: placed {placed} "
                f"for requested {requested}"), detail
    return None, detail


def published_raw_criterion(track, structure, motif_map, context):
    """The ONE published criterion measured on the raw design, not on the refold.

    enzyme success.txt criterion 2: "No ligand clashes. A ligand clash is any non-motif
    backbone atom (N, CA or C) within 1.5 A of the closest ligand atom." No other
    track's success.txt names a term on the raw backbone; they are all measured on the
    refolded model, which `pipeline.DesignabilityScorer` owns.

    Checked before folding purely to save the fold, never to add a requirement: it is a
    conjunct of the criterion, so the verdict is the same either way. It rejected 52% of
    enzyme designs on the 2026-09-26 reference run (86 of the 164 that view then held),
    which is most of the grading cost this track would otherwise spend on designs that
    cannot pass.
    """
    detail = {}
    if track != "enzyme":
        return None, detail
    coordinates, _, _ = structure
    ligand = context.get("ligand_coords")
    if ligand is None or np.size(ligand) == 0:
        raise ValueError(f"{track} context is missing ligand coordinates")
    placed = set(motif_map or ())
    keep = np.array([index not in placed for index in range(coordinates.shape[0])])
    backbone = coordinates[keep][:, :3].reshape(-1, 3) if keep.any() else np.empty((0, 3))
    detail["ligand_min_distance"] = minimum_distance(backbone, ligand)
    if detail["ligand_min_distance"] < LIGAND_CLASH_MIN:
        return (f"non-motif backbone clashes with ligand at "
                f"{detail['ligand_min_distance']:.2f} A"), detail
    return None, detail


def fold_designs(scorer, track, context, to_fold):
    """Score `to_fold` through the predictor, `scorer.devices` designs at a time.

    Returns {order -> verdict or the exception it raised}, so the caller can replay them
    in submission order. Threads are the right tool despite the GIL: every worker spends
    its time waiting on a `subprocess.run` for LigandMPNN and RoseTTAFold3, and
    `DesignabilityScorer.evaluate` leases a distinct GPU per call.

    One worker is the degenerate case and takes the direct path, so a single-GPU run
    behaves exactly as it did before and needs no thread pool at all.
    """
    workers = min(len(getattr(scorer, "devices", [None])), len(to_fold)) or 1
    if workers == 1:
        return {order: scorer.evaluate(track, structure, context, motif_map)
                for order, structure, motif_map in to_fold}

    verdicts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scorer.evaluate, track, structure, context, motif_map): order
            for order, structure, motif_map in to_fold}
        for future in concurrent.futures.as_completed(futures):
            order = futures[future]
            try:
                verdicts[order] = future.result()
            except BaseException as error:               # noqa: BLE001 - re-raised in order
                # A scorer that cannot run is infrastructure failure, never a zero score
                # (see the module docstring), so it must reach the caller. Carrying it
                # instead of raising here keeps WHICH design failed deterministic.
                verdicts[order] = error
    return verdicts


def score_track(track, sealed_path, requests, context_root, scorer, novelty=None):
    """Score every requested design for one track.

    With `novelty` (novelty.FitNovelty) a designable design passes only if its maximum TM-score to the fit
    corpus is below NOVELTY_MAX_TM; diversity is then measured over those passing designs."""
    outcome = {"designs_requested": 0, "designs_returned": 0, "designs_valid": 0,
               "designs_designable": 0, "conditioning_satisfied": 0, "failures": {}}
    if novelty is not None:
        outcome["designs_novel"] = 0
    try:
        submitted = {}
        for row in jsonl(sealed_path):
            if (set(row) != {"example_id", "designs"}
                    or row["example_id"] not in requests
                    or row["example_id"] in submitted
                    or not isinstance(row["designs"], list)):
                raise ValueError("malformed, unknown, or duplicate design record")
            submitted[row["example_id"]] = row["designs"]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"pass_rate": 0.0, "diversity": 0.0,
                "invalid_reason": str(error), **outcome}

    per_target_success, per_target_diversity = [], []
    structures_root = sealed_path.parent / "structures"
    designable = {}          # example_id -> [(order, CA trace, PDB path)], in submission order

    for example_id, request in sorted(requests.items()):
        requested = request["num_designs"]
        outcome["designs_requested"] += requested
        context = load_context(Path(context_root) / request["context_path"])
        designs = submitted.get(example_id, [])[:requested]
        outcome["designs_returned"] += len(designs)
        paths = {}

        # Pass 1, sequential and cheap: parse, validate the request, and apply the one
        # published criterion measured on the raw design. Anything that survives has to
        # be folded, which is the only expensive part.
        to_fold = []
        for order, design in enumerate(designs):
            if (not isinstance(design, dict) or not set(design) <= DESIGN_FIELDS
                    or "structure_path" not in design):
                outcome["failures"]["malformed_design_record"] = \
                    outcome["failures"].get("malformed_design_record", 0) + 1
                continue
            try:
                path = (structures_root / design["structure_path"]).resolve()
                if not path.is_relative_to(structures_root.resolve()):
                    raise ValueError("structure path escapes the sealed snapshot")
                structure = parse_allatom_pdb(path, request["design_length"])
                outcome["designs_valid"] += 1
                motif = design.get("motif_map")
                reason, _ = validate_request(track, structure, motif, context)
                if reason is None:
                    outcome["conditioning_satisfied"] += 1
                    reason, _ = published_raw_criterion(track, structure, motif, context)
            except (OSError, ValueError) as error:
                outcome["failures"][str(error)[:80]] = \
                    outcome["failures"].get(str(error)[:80], 0) + 1
                continue
            if reason:
                outcome["failures"][reason[:80]] = outcome["failures"].get(reason[:80], 0) + 1
                continue
            motif_map = design.get("motif_map") if track in MOTIF_TRACKS else None
            to_fold.append((order, structure, motif_map))
            paths[order] = path

        # Pass 2, parallel: the pipeline owns the criterion, because the predictor is the
        # only thing that can judge it. The grader adds NOTHING to it - no geometry gate,
        # no motif RMSD on the raw design, no ligand burial, no hotspot contact. Those
        # were this task's own inventions and they deleted most of the signal: 59% of
        # designable enzyme designs and every designable ppi binder. See RESULTS.md.
        #
        # One design per worker, one GPU per worker. Designs are independent and every
        # fold is already its own subprocess, so this is the whole cost of using more
        # than one card. Parallelism is confined to HERE: the verdicts are collected and
        # then replayed in submission order below, because `cluster_designs` is a greedy
        # walk whose cluster count depends on the order it sees backbones in. A pass rate
        # that changed with scheduling would not be reproducible.
        verdicts = fold_designs(scorer, track, context, to_fold)

        for order, structure, _ in to_fold:
            verdict = verdicts[order]
            if isinstance(verdict, BaseException):
                raise verdict
            if verdict.get("passed"):
                outcome["designs_designable"] += 1
                designable.setdefault(example_id, []).append(
                    (order, structure[0][:, 1], paths[order]))     # CA trace
            else:
                parts = [f"scRMSD {verdict['sc_rmsd']:.2f}"] if np.isfinite(
                    verdict.get("sc_rmsd", float("inf"))) else []
                for label, key in (("ipTM", "iptm"), ("PAE", "interface_pae"),
                                   ("ligRMSD", "ligand_rmsd"), ("motif", "motif_rmsd")):
                    value = verdict.get(key)
                    if isinstance(value, (int, float)):
                        parts.append(f"{label} {value:.2f}")
                # Keep the stated reason alongside the numbers. Dropping it whenever any
                # numeric field was present hid which term actually failed - the metrics
                # reported are the best over all sequences, so they need not be the ones
                # that lost.
                if verdict.get("reason"):
                    parts.append(str(verdict["reason"]))
                key = ("not designable (" + ", ".join(parts) + ")" if parts
                       else "not designable")
                outcome["failures"][key] = outcome["failures"].get(key, 0) + 1

    # Novelty, batched over the track: one Foldseek search of every designable design against the fit corpus.
    max_tm = {}
    if novelty is not None:
        max_tm = novelty.max_tm({(example_id, order): path for example_id, rows in designable.items()
                                 for order, _, path in rows})
        if max_tm:
            outcome["novelty_max_tm_mean"] = float(np.mean(list(max_tm.values())))
    for example_id, request in sorted(requests.items()):
        passing = []
        for order, trace, _ in designable.get(example_id, []):
            score = max_tm.get((example_id, order))
            if score is not None and score >= NOVELTY_MAX_TM:
                key = f"near-copy of a fit-corpus structure (max TM >= {NOVELTY_MAX_TM})"
                outcome["failures"][key] = outcome["failures"].get(key, 0) + 1
                continue
            passing.append(trace)
        if novelty is not None:
            outcome["designs_novel"] += len(passing)
        per_target_success.append(len(passing) / request["num_designs"])
        per_target_diversity.append(cluster_designs(passing) / len(passing) if passing else 0.0)

    pass_rate = float(np.mean(per_target_success)) if per_target_success else 0.0
    diversity = float(np.mean(per_target_diversity)) if per_target_diversity else 0.0
    # Reported side by side, never blended. Mode coverage answers a different question
    # from foldability, and folding the two into one number would hide which moved.
    # On `unconditional` diversity is the anti-degeneracy readout: a generator emitting
    # one designable fold repeatedly reaches pass_rate 1.0 with diversity near zero.
    outcome.update({"pass_rate": pass_rate, "diversity": diversity})
    outcome["invalid_reason"] = None
    return outcome


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in ("invalid_reason", "failures"):
                flatten(f"{prefix}_{key}" if prefix else key, item, output)
    elif type(value) in (int, float) and math.isfinite(value):
        output[prefix] = float(value)


def main(scorer=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--contexts", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    # The image sets both of these, so the defaults are right inside a container. They
    # are overridable because the grader also has to run OUTSIDE one - reproducing a
    # known answer on another cluster is the only way to check that a build scores the
    # same. score.py and benchmark/designability/run_designability.py already read these
    # env vars, and pipeline.py's own "not installed" message tells the operator to set
    # LIGANDMPNN_ROOT; until this was added, the trusted grader was the one component
    # that ignored it.
    parser.add_argument("--assets", default=os.environ.get(
        "PROTEIN_DESIGN_ASSETS", "/assets"),
        help="LigandMPNN + RoseTTAFold3 parameters; see "
             "environment/fetch_designability_weights.sh")
    parser.add_argument("--ligandmpnn", default=os.environ.get(
        "LIGANDMPNN_ROOT", "/opt/ligandmpnn"),
        help="LigandMPNN checkout (the vendored environment/evaluation/ligandmpnn)")
    parser.add_argument("--reference", default=os.environ.get(
        "PROTEIN_DESIGN_NOVELTY_REFERENCE", "/reference/fit"),
        help="fit-corpus structures for the novelty gate (development/fit/structures, read-only)")
    parser.add_argument("--wait-seconds", type=int, default=129600)
    # One design folded per worker, one GPU leased per worker. The default is however
    # many devices are visible, so a 1-GPU host behaves exactly as it did before and an
    # 8-GPU verifier uses all eight without being told to. Set it to 1 to force the
    # serial path.
    parser.add_argument("--workers", type=int, default=None,
                        help="designs to fold concurrently (default: visible GPUs)")
    arguments = parser.parse_args()

    ready = arguments.sealed.parent / "READY"
    deadline = time.monotonic() + arguments.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("design snapshot was not published")
        time.sleep(1)

    snapshot_path = arguments.sealed / "snapshot.json"
    if ready.read_text().strip() != sha256(snapshot_path):
        raise ValueError("snapshot ready hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    if (set(snapshot) != {"schema", "input_manifest_sha256", "files", "structures"}
            or snapshot["schema"] != SNAPSHOT_SCHEMA):
        raise ValueError("invalid snapshot schema")
    for name, digest in snapshot["files"].items():
        if sha256(arguments.sealed / name) != digest:
            raise ValueError(f"sealed design hash mismatch: {name}")

    from common import validate_input
    _, requests = validate_input(arguments.contexts)

    if scorer is None:
        from pipeline import DesignabilityScorer, resolve_devices
        devices = resolve_devices()
        if arguments.workers:
            # Fewer workers than devices: keep the first N. More workers than devices:
            # round-robin them, which is legitimate only if the GPUs have memory for it.
            devices = [devices[index % len(devices)]
                       for index in range(arguments.workers)]
        scorer = DesignabilityScorer(assets_root=arguments.assets,
                                     ligandmpnn_root=arguments.ligandmpnn,
                                     devices=devices)
        print(f"grading with {len(devices)} worker(s) on device(s) "
              f"{', '.join(str(d) for d in devices)}", flush=True)

    from novelty import FitNovelty
    novelty = FitNovelty(arguments.reference)
    details = {track: score_track(track, arguments.sealed / f"{track}.jsonl",
                                  requests[track], arguments.contexts, scorer, novelty)
               for track in TRACKS}

    arguments.output.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output / "report.json", {
        "schema": REPORT_SCHEMA,
        "thresholds": {"sc_rmsd_max": SC_RMSD_MAX, "plddt_min": PLDDT_MIN,
                       "motif_rmsd_max": MOTIF_RMSD_MAX, "novelty_max_tm": NOVELTY_MAX_TM},
        "tracks": details,
    })
    # reward.json is the file CONTRIBUTING requires; it carries the per-track numeric
    # metrics. No composite key: see the module docstring.
    flat = {}
    for track, result in details.items():
        flatten(track, result, flat)
    write_json(arguments.output / "reward.json", harbor_reward(flat))


if __name__ == "__main__":
    try:
        main()
    except ScorerUnavailable as error:
        # Infrastructure failure: exit nonzero so it cannot be read as a measured score.
        raise SystemExit(f"designability scorer unavailable: {error}")
