"""Build/check the public-resource selection. Never certify unrun CFD cases.

All inputs are checked-in small metadata, not mutable Hub main or host paths.
The output is a data release; suite.json separately binds validated executors.
"""

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parent
RELEASE_ID = "hydrogym-public-500gb-v1"
REVISION = "cf5ef6748443ae72e9647f398487e98d421a7243"
CODE_REVISION = "4ab9854dea3d84e38a59c25e0f5835a00cf8225f"
# User-approved membership, not a size-sorted automatic subset. Keep every
# selected case's published inputs. A changed selection requires a new ID.
SELECTED_CASE_IDS = frozenset("""
Cavity_2D_Re7500
Cavity_2D_Re7500_fine_FD
Cavity_2D_Re7500_medium_FD
Channel_3D_Retau180
Cube_3D_Re3700
Cylinder_2D_Re1000
Cylinder_2D_Re100_fine_FD
Cylinder_2D_Re100_medium_FD
Cylinder_2D_Re200
Cylinder_2D_Re200_fine_FD
Cylinder_2D_Re200_medium_FD
Cylinder_2D_Re3900
Cylinder_3D_Re200
Cylinder_3D_Re3900
DRA2303Jet_3D_Re400000_Ma02
Kolmogorov_2D
NACA0012Gust_2D_Re1000_AOA20
NACA0012Gust_2D_Re1000_AOA40
NACA0012Gust_2D_Re100_AOA20
NACA0012Gust_2D_Re100_AOA40
NACA0012Gust_3D_Re10000_AOA12
NACA0012Gust_3D_Re50000_AOA12
NACA0012_2D_Re1000_AOA20
NACA0012_2D_Re1000_AOA40
NACA0012_2D_Re100_AOA20
NACA0012_2D_Re100_AOA40
NACA0012_3D_Re10000_AOA12
NACA0012_3D_Re50000_AOA12
Nozzle2D_coarse
Nozzle2D_fine
Nozzle3D_coarse
Pinball_2D_Re100
Pinball_2D_Re100_fine_FD
Pinball_2D_Re150
Pinball_2D_Re150_fine_FD
Pinball_2D_Re30
Pinball_2D_Re30_fine_FD
Pinball_2D_Re75
Pinball_3D_Re100
Pinball_3D_Re150
Pinball_3D_Re30
Pinball_3D_Re75
Sphere_3D_Re300
Sphere_3D_Re3700
SquareCylinder_2D_Re1000
SquareCylinder_2D_Re200
TCFmini_3D_Re180
""".split())
DATA_BUDGET = {
    "max_bytes": 500_000_000_000,
    "scope": "One shared copy of fixed training/evaluation source assets",
    "excludes": ["container_images", "generated_trajectories", "model_checkpoints",
                 "logs", "solver_scratch", "download_caches"],
}
# Required inputs from pinned hydrogym/data_manager.py SOLVER_PROFILES.
# Add actual JAX-Fluids inputs and native MAIA YAML/reward inputs, because its
# permissive generic downloader check alone does not establish completeness.
PROFILES = {
    ".MAIA_LB": ("maia", ["environment_config.yaml", "geometry.toml",
        "properties_grid.toml", "properties_run.toml", "grid.Netcdf",
        "restart_.Netcdf", "stl/*"]),
    ".MAIA_STRCTRD": ("maia-structured", ["environment_config.yaml",
        "properties_run.toml", "grid.hdf5", "restart_.hdf5"]),
    ".JAX": ("jax", ["environment_config.yaml", "properties_run.toml", "initial_field/*"]),
    ".JAXFLUIDS": ("jax-fluids", ["environment_config.yaml", "jxf_case_setup.json",
        "jxf_numerical_setup.json", "restart.h5"]),
    ".NEK5000": ("nek5000-v19", ["environment_config.yaml", "int_pos",
        "*.re2", "*.ma2", "*.par", "restart_files/*"]),
    # Native NEK_INIT.get_Case_Files for v17 uses .map, not .ma2. .sch is
    # disposable runtime state removed at startup, not an input dataset.
    ".NEK5000_v17": ("nek5000-v17", ["environment_config.yaml", "stat_pts.in",
        "forparam.i", "*.re2", "*.rea", "*.map", "*.wall",
        "*.restart", "mask_*", "restart_files/*"]),
    ".FIREDRAKE": ("firedrake", ["*.ckpt"]),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_index(index):
    if index.get("revision") != REVISION:
        raise ValueError("Wrong source index revision")
    paths = set()
    for item in index["files"]:
        name = item["rfilename"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name in paths:
            raise ValueError(f"Unsafe/duplicate source file: {name}")
        paths.add(name)
        if type(item.get("size")) is not int or item["size"] < 0:
            raise ValueError(f"Invalid source size: {name}")
        checksum = item.get("lfs", {}).get("sha256", item.get("blobId", ""))
        length = 64 if item.get("lfs") else 40
        if not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", checksum):
            raise ValueError(f"Invalid source hash: {name}")


def resource_inputs(name, files, config=None):
    relative = {f["rfilename"].removeprefix(name + "/"): f for f in files}
    markers = [marker for marker in PROFILES if marker in relative]
    # Older public folders legitimately have no marker. Match their explicit
    # YAML section instead, as upstream's backend-specific fallback does.
    if not markers and config:
        if "maia" in config:
            markers = [".MAIA_LB"]
        elif "jax" in config:
            markers = [".JAX"]
        elif "jaxfluids" in config:
            markers = [".JAXFLUIDS"]
    if len(markers) != 1:
        return None, ["Exactly one known solver marker is required"]
    backend, patterns = PROFILES[markers[0]]
    patterns = list(patterns)
    if name == "Kolmogorov_2D":
        # Pinned KolmogorovFlow.initialize_state generates its analytic state.
        patterns.remove("initial_field/*")
    elif name == "Channel_3D_Retau180":
        patterns.remove("initial_field/*")
        patterns.extend(["initial_field/U.npy", "initial_field/V.npy", "initial_field/W.npy"])
    elif name == "Nozzle3D_fine":
        # The public restart is 3 host files, each with 4 x-domain shards.
        # Do not misclassify a complete distributed state as missing data just
        # because build_tvc_runtime_setup currently expects restart.h5.
        patterns.remove("restart.h5")
        patterns.extend([f"restart_proc{i}.h5" for i in range(3)])
    if name.startswith("Cavity") and backend == "maia":
        patterns.append("baseline_state.feather")
    missing = [p for p in patterns if not any(
        fnmatch.fnmatchcase(n, p) and f["size"] > 0 for n, f in relative.items())]
    return backend, missing


def build(audit, index, native_configs, *, selected_case_ids=None):
    selected_case_ids = SELECTED_CASE_IDS if selected_case_ids is None else frozenset(selected_case_ids)
    validate_index(index)
    if audit["source_revision"] != REVISION or native_configs["revision"] != REVISION:
        raise ValueError("Inconsistent frozen source revisions")
    configs = {r["name"]: r["config"] for r in native_configs["configs"]}
    by_name = {}
    for item in index["files"]:
        if "/" in item["rfilename"]:
            by_name.setdefault(item["rfilename"].split("/")[0], []).append(item)
    mapping, exclusions = {}, []
    for row in audit["rows"]:
        if not row["candidate_payloads"]:
            exclusions.append({"paper_row": row["paper_row"], "family": row["family"],
                "dimension": row["dimension"], "parameters": row["parameters"],
                "reason": row["resource_status"], "detail": row["notes"]})
            continue
        if row["family"] == "cylinder-marl":
            exclusions.append({"paper_row": row["paper_row"], "family": row["family"],
                "dimension": row["dimension"], "parameters": row["parameters"],
                "reason": "missing_distinct_multiagent_configuration",
                "detail": "Candidate YAML declares one agent; reusing its mesh does not define the paper MARL task. Single-agent counterparts remain included."})
            continue
        for name in row["candidate_payloads"]:
            mapping.setdefault(name, []).append(row)
        for name in row["marker_only"]:
            exclusions.append({"environment": name, "paper_row": row["paper_row"],
                "reason": "marker_only_actuation_variant", "detail": "No nonempty CFD payload."})
    cases = []
    for name, rows in sorted(mapping.items()):
        files = sorted(by_name.get(name, []), key=lambda f: f["rfilename"])
        backend, missing = resource_inputs(name, files, configs.get(name))
        if missing:
            exclusions.append({"environment": name,
                "paper_rows": [r["paper_row"] for r in rows],
                "reason": "incomplete_native_resource_package", "missing": missing,
                "detail": "Required by the pinned solver's data profile; not excluded for model performance or lack of our evaluator adapter. Can be reconsidered after a validated repair in a new release."})
            continue
        cfg = configs.get(name)
        reconstructed = False
        if backend == "firedrake":
            match = re.fullmatch(r"(Cylinder|Pinball|Cavity)_2D_Re(\d+)_(fine|medium)_FD", name)
            if match is None:
                raise ValueError(f"Unreviewed Firedrake identity: {name}")
            # Expose the identity, do not invent a time step/actuator/obs recipe.
            cfg = {"flow_family": match[1], "Re": int(match[2]), "mesh": match[3],
                   "configuration_source": "resource name plus pinned Firedrake flow implementation",
                   "checkpoint_files": [f["rfilename"] for f in files if f["rfilename"].endswith(".ckpt")]}
            reconstructed = True
        elif cfg is None:
            raise ValueError(f"Missing captured native configuration: {name}")
        row = rows[0]
        cases.append({"id": name, "environment": name, "family": row["family"],
            "dimension": row["dimension"], "parameters": row["parameters"],
            "backend": backend, "paper_rows": [r["paper_row"] for r in rows],
            "files_sha256": digest(files), "file_count": len(files),
            "bytes": sum(f["size"] for f in files), "native_configuration": cfg,
            "configuration_identity_reconstructed": reconstructed,
            "resource_status": "required_payloads_present",
            "execution_status": "full_release_replay_pending"})
        if name == "Nozzle3D_fine":
            cases[-1]["packaging_note"] = "Complete 12-domain-shard restart in 3 files. Native helper expects restart.h5; a lossless layout adapter is required, not fabricated flow."
        if backend == "nek5000-v17":
            cases[-1]["packaging_note"] = "Native v17 requires .map and removes .sch at startup. Generic data-manager .ma2/.sch requirements are incompatible and need an adapter correction."
    available_ids = {c["id"] for c in cases}
    if selected_case_ids - available_ids:
        raise ValueError(f"Selected cases lack complete resources: {sorted(selected_case_ids - available_ids)}")
    for case in cases:
        if case["id"] not in selected_case_ids:
            exclusions.append({"environment": case["environment"],
                "family": case["family"], "dimension": case["dimension"],
                "parameters": case["parameters"], "backend": case["backend"],
                "bytes": case["bytes"], "paper_rows": case["paper_rows"],
                "reason": "not_selected_500gb_scope",
                "resource_status": "required_payloads_present",
                "detail": "User-approved 500 GB scope selection, not missing data or a policy-performance exclusion. Preserve all 2D cases; paired 3D cylinder/sphere Reynolds numbers; all four 3D Pinball cases; cube Re3700; NACA Re10000/50000 at AOA12 with both objectives; both channels; coarse 3D nozzle; DRA2303 jet Ma0.2."})
    cases = [c for c in cases if c["id"] in selected_case_ids]
    ids = [c["id"] for c in cases]
    selected_files = [f for f in index["files"] if f["rfilename"].split("/")[0] in set(ids)]
    total_bytes = sum(f["size"] for f in selected_files)
    if total_bytes > DATA_BUDGET["max_bytes"]:
        raise ValueError("Selected source assets exceed the frozen 500 GB data budget")
    return {"release_id": RELEASE_ID, "selection_status": "frozen",
        "status": "data-release-not-cfd-certified", "created": "2026-09-06",
        "scope": "User-approved 500 GB subset of resource-complete public variants matching the audited paper tables. Not the original full paper benchmark; solver/fidelity/objective variants are separate cases.",
        "supersedes": "hydrogym-public-resources-v1",
        "data_budget": DATA_BUDGET,
        "source": {"repo_id": index["repo_id"], "revision": REVISION,
                   "code_repo": "https://github.com/dynamicslab/hydrogym",
                   "code_revision": CODE_REVISION},
        "source_index_sha256": digest(index),
        "contract": "hydrogym-policy/v1",
        "case_count": len(cases), "backend_counts": dict(sorted(Counter(c["backend"] for c in cases).items())),
        "files": {"count": len(selected_files), "bytes": total_bytes},
        "training": {"kind": "interactive-environments", "case_ids": ids,
            "initial_conditions": "All supplied initial states belonging to included cases",
            "external_data": False, "fixed_transition_count": None},
        "test": {"case_ids": ids, "regime": "known-task-frozen-model",
            "visible": True, "independent_hidden_split": False,
            "one_shared_checkpoint": True, "parameter_updates": False,
            "normalizer_updates": False, "initial_state_pool_shared_with_training": True,
            "primary_metric": "mean_undiscounted_native_episode_return_per_case",
            "aggregate_metric": "none_until_versioned_same_protocol_reference_anchors_exist",
            "execution_manifest": "suite.json",
            "execution_manifest_status": "requires_native_protocol_binding_and_full_replay"},
        "cases": cases, "exclusions": exclusions,
        "reconstruction_policy": "No missing geometry/flow is synthesized in v1. Firedrake identity is decoded from public checkpoint directory names; its native executable binding still requires validation. Any later repair changes the release ID; never silently backfill a frozen test."}


def load_release(path=None):
    value = json.loads(Path(path or ROOT / "release.json").read_text())
    if value.get("release_id") == "hydrogym-public-research-v2":
        from research_release import build_current
        if value != build_current():
            raise ValueError("Research release differs from its exact frozen selection/evidence/source metadata")
        return value
    if value.get("release_id") == "hydrogym-public-hour-v1":
        from hour_release import build_current
        if value != build_current():
            raise ValueError("Hour release differs from its exact frozen training evidence/source metadata")
        return value
    if value.get("release_id") == "hydrogym-public-coverage-v1":
        from coverage_release import build_current
        if value != build_current():
            raise ValueError("Coverage release differs from its exact frozen selection/evidence/source metadata")
        return value
    if value.get("release_id") == "hydrogym-public-speed-v1":
        from speed_release import build_current
        if value != build_current():
            raise ValueError("Speed release differs from its exact frozen selection/evidence/source metadata")
        return value
    ids = [c["id"] for c in value["cases"]]
    if value.get("release_id") != RELEASE_ID or value.get("selection_status") != "frozen":
        raise ValueError("Unknown/unfrozen public release")
    if not ids or len(set(ids)) != len(ids) or value["case_count"] != len(ids):
        raise ValueError("Invalid public release case list")
    if value["training"]["case_ids"] != ids or value["test"]["case_ids"] != ids:
        raise ValueError("Train/test selection drift")
    if ids != sorted(SELECTED_CASE_IDS):
        raise ValueError("Public release differs from the exact frozen 47-case selection")
    if value.get("data_budget") != DATA_BUDGET:
        raise ValueError("Wrong frozen data budget")
    totals = {"count": sum(c["file_count"] for c in value["cases"]),
              "bytes": sum(c["bytes"] for c in value["cases"])}
    if value["files"] != totals or totals["bytes"] > DATA_BUDGET["max_bytes"]:
        raise ValueError("Invalid source totals or exceeded data budget")
    if value["source"]["revision"] != REVISION:
        raise ValueError("Wrong public resource revision")
    return value


def validate_suite_binding(release, suite):
    if suite.get("data_release_id") != release["release_id"] or suite.get("data_release_sha256") != digest(release):
        raise ValueError("Evaluation suite is not bound to the frozen data release")
    cases = suite.get("cases", [])
    indexed = {c["id"]: c for c in cases}
    if len(indexed) != len(cases) or set(indexed) != set(release["test"]["case_ids"]):
        raise ValueError("Evaluation must cover the exact frozen test case list")
    for case in release["cases"]:
        if indexed[case["id"]].get("environment") != case["environment"]:
            raise ValueError("Evaluation replaced a frozen environment")
        if indexed[case["id"]].get("backend") != case["backend"]:
            raise ValueError("Evaluation replaced a frozen backend")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the stored release against its source metadata")
    parser.add_argument("--legacy", action="store_true", help="Rebuild/check the archived 47-case storage-only release")
    args = parser.parse_args()
    value = build(json.loads((ROOT / "resource-audit.json").read_text()),
        json.loads((ROOT / "source-files.json").read_text()),
        json.loads((ROOT / "native-configs.json").read_text()))
    stored = ROOT / "release-500gb-v1.json" if args.legacy else ROOT / "release.json"
    if not args.legacy and json.loads(stored.read_text()).get("release_id") == "hydrogym-public-speed-v1":
        from speed_release import build_current
        value = build_current()
    elif not args.legacy and json.loads(stored.read_text()).get("release_id") == "hydrogym-public-coverage-v1":
        from coverage_release import build_current
        value = build_current()
    elif not args.legacy and json.loads(stored.read_text()).get("release_id") == "hydrogym-public-hour-v1":
        from hour_release import build_current
        value = build_current()
    elif not args.legacy and json.loads(stored.read_text()).get("release_id") == "hydrogym-public-research-v2":
        from research_release import build_current
        value = build_current()
    if args.check:
        if load_release(stored) != value:
            raise ValueError("Stored release differs from frozen source metadata")
        print(json.dumps({"release_id": value["release_id"], "case_count": value["case_count"],
                          "sha256": digest(value), "backend_counts": value["backend_counts"]}))
    else:
        print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
