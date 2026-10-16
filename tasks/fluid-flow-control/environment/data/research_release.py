"""Freeze the approved Research v2 data selection, not unrun CFD validation.

Reuses the existing file inventory and materializer. All historical releases
remain loadable; cost classes never make expensive scoring cases optional.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import hour_release
import public_release as public

ROOT = Path(__file__).resolve().parent
RELEASE_ID = "hydrogym-public-research-v2"
SELECTION_SHA256 = "b3a2e1e7498a368083776a039d73d0eef4e5de1c46116ec71d4276cf78bd6847"
TIER_COUNTS = {"core": 30, "borderline": 5,
               "single_gpu_exception": 1, "multigpu_exception": 3}


def read(name):
    return json.loads((ROOT / name).read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def build(selection, evidence, prior):
    if (selection.get("selection_id") != "hydrogym-research-v2"
            or selection.get("status") != "user-approved-frozen-selection"
            or selection["policy"]["max_source_bytes"] != 500_000_000_000):
        raise ValueError("Unknown/unapproved Research v2 selection or budget")
    if (evidence["source_revision"] != public.REVISION
            or evidence["code_revision"] != public.CODE_REVISION):
        raise ValueError("Research source/code evidence mismatch")
    decisions = {d["case"]: d for d in selection["cases"]}
    ids = sorted(decisions)
    if len(ids) != 39 or len(ids) != len(selection["cases"]):
        raise ValueError("Research v2 requires 39 unique cases")
    contract = selection["contract"]
    if (contract["training_case_ids"] != ids or contract["evaluation_case_ids"] != ids
            or contract["one_jointly_trained_shared_checkpoint"] is not True
            or contract["all_selected_cases_required"] is not True
            or contract["evaluation_parameter_updates"] is not False
            or contract["evaluation_normalizer_updates"] is not False):
        raise ValueError("Research training/test/shared-checkpoint contract drift")
    result = public.build(read("resource-audit.json"), read("source-files.json"),
                          read("native-configs.json"), selected_case_ids=ids)
    if result["files"] != {"count": 558, "bytes": 453_756_315_982}:
        raise ValueError("Research source file membership or total changed")
    previous = {c["id"]: c for c in prior["cases"]}
    if set(previous) - set(ids):
        raise ValueError("Research v2 must preserve all hour-v1 cases")
    records = {r["case"]: r for r in evidence["cases"]}
    if len(records) != len(evidence["cases"]):
        raise ValueError("Duplicate cost evidence")
    wall_hours = gpu_hours = 0.0
    for case in result["cases"]:
        name = case["id"]
        d, r = decisions[name], records[name]
        samples = r.get("timing_samples_seconds", [])
        if (r.get("status") != "complete" or r.get("container_exit_code") != 0
                or len(samples) < 2 or not all(positive(x) for x in samples)
                or not positive(r.get("reset_seconds"))
                or not positive(r.get("startup_seconds"))
                or r.get("observed_field_response") is False):
            raise ValueError("Missing valid native short-loop evidence: " + name)
        if (r["measured_controls"] != len(samples)
                or not math.isclose(sum(samples) / len(samples), r["seconds_per_control"], rel_tol=1e-10)):
            raise ValueError("Timing samples differ from summary: " + name)
        if (d["source_bytes"] != case["bytes"] or r["data_bytes"] != case["bytes"]
                or d["backend"] != case["backend"] or r["backend"] != case["backend"]
                or d["dimension"] != case["dimension"] or r["dimension"] != case["dimension"]
                or d["hardware"] != r["hardware"]):
            raise ValueError("Source or hardware substitution: " + name)
        gpus = {"1 H100": 1, "2 H100": 2}.get(r["hardware"])
        if gpus is None or d["single_gpu_cost_established"] != (gpus == 1):
            raise ValueError("Unknown or disguised GPU cost: " + name)
        if gpus == 2:
            bindings = r.get("gpu_bindings", [])
            if (len(bindings) != 2 or len({b["cuda_device_uuid"] for b in bindings}) != 2
                    or any(b["visible_cuda_devices"] != 1 for b in bindings)):
                raise ValueError("Missing distinct two-GPU rank bindings: " + name)
        if d["reference_training_controls"] != 4000 or d["full_4000_step_training_validated"] is not False:
            raise ValueError("Reference budget is not completed training: " + name)
        horizon = (201 if case["backend"] == "maia" else
                   100 if name == "Channel_3D_Retau180" else
                   1000 if name == "Kolmogorov_2D" else 667)
        if d["native_episode_controls_for_costing"] != horizon:
            raise ValueError("Changed costing horizon: " + name)
        if name in previous:
            old = previous[name]
            budget = hour_release.estimate(old["training_cost_evidence"], 4000)
            base_hours = budget["estimated_seconds"] / 3600
            profile = old["runtime_profile"]
            case["execution_status"] = old["execution_status"]
            case["training_evidence_reference"] = {"file": "cost-evidence-hour-v1.json", "case": name}
        else:
            seconds = (r["startup_seconds"] + 4000 * r["seconds_per_control"]
                       + math.ceil(4000 / horizon) * r["reset_seconds"]
                       + 2 * horizon * r["seconds_per_control"] + 300)
            base_hours = seconds / 3600
            profile = r["effective_diagnostic_configuration"]
            case["execution_status"] = "short_loop_passed_full_evaluation_and_training_pending"
        planning_hours = base_hours * 1.25
        if (not math.isclose(base_hours, d["whole_experiment_estimated_base_hours"], rel_tol=1e-10)
                or not math.isclose(planning_hours, d["whole_experiment_planning_hours"], rel_tol=1e-10)):
            raise ValueError("Cost estimate changed: " + name)
        tier = ("multigpu_exception" if gpus == 2 else "core" if planning_hours <= 50
                else "borderline" if planning_hours <= 70 else "single_gpu_exception")
        if d["tier"] != tier or not d["scientific_value"]:
            raise ValueError("Cost exception or scientific value missing: " + name)
        case.update(runtime_profile=copy.deepcopy(profile),
                    runtime_profile_status="cost-profile-not-full-evaluator-certification",
                    selection_review=copy.deepcopy(d),
                    cost_evidence_reference={"file": "cost-evidence-coverage-v1.json", "case": name},
                    required_solver_gpus=gpus)
        wall_hours += planning_hours
        gpu_hours += planning_hours * gpus
    if Counter(d["tier"] for d in decisions.values()) != TIER_COUNTS:
        raise ValueError("Research v2 cost-tier membership changed")
    excluded = {d["case"]: d for d in selection["excluded_complete_resources"]}
    scope_exclusions = [e for e in result["exclusions"] if e["reason"] == "not_selected_500gb_scope"]
    if (len(excluded) != 23 or len(excluded) != len(selection["excluded_complete_resources"])
            or set(excluded) != {e["environment"] for e in scope_exclusions}):
        raise ValueError("Every complete unselected resource needs a unique decision")
    for case in scope_exclusions:
        d = excluded[case["environment"]]
        if d["source_bytes"] != case["bytes"] or not d["reason"] or not d["detail"]:
            raise ValueError("Exclusion evidence missing")
        case.update(reason=d["reason"], detail=d["detail"])
    totals = selection["totals"]
    if (totals["case_count"] != 39 or totals["two_dimensional_cases"] != 20
            or totals["three_dimensional_cases"] != 19
            or sum(c["dimension"] == 3 for c in result["cases"]) != 19
            or totals["source_bytes"] != result["files"]["bytes"]
            or totals["counts_by_tier"] != TIER_COUNTS
            or not math.isclose(totals["serial_wall_hours_on_each_cases_stated_hardware"], wall_hours, rel_tol=1e-10)
            or not math.isclose(totals["total_estimated_gpu_hours"], gpu_hours, rel_tol=1e-10)):
        raise ValueError("Research totals changed")
    result.update(release_id=RELEASE_ID, created="2026-09-08",
                  supersedes=hour_release.RELEASE_ID,
                  scope="User-approved known-task Research v2: 39 cases, 20 in 2D and 19 in 3D. Not the complete paper benchmark.",
                  selection_spec_sha256=public.digest(selection),
                  cost_evidence_sha256=public.digest(evidence),
                  selection_policy=copy.deepcopy(selection["policy"]),
                  reference_training_budget={"controls_per_case": 4000, "total_controls": 156000,
                      "costing": copy.deepcopy(selection["costing"]), "totals": copy.deepcopy(totals),
                      "paper_convergence_claim": False, "agent_training_limit": False},
                  reconstruction_policy="All source bytes, grids, dt and native horizons unchanged. Explicit cost profiles and the validated Channel actuator-clock repair are preserved. Formal evaluator bindings and full 39-case replay remain pending.")
    result["source"]["organizer_runtime_adapters"] = copy.deepcopy(prior["source"]["organizer_runtime_adapters"])
    result["test"]["all_selected_cases_required"] = True
    result["test"]["cost_tiers_are_optional"] = False
    return result


def build_current():
    path = ROOT / "selection-research-v2.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SELECTION_SHA256:
        raise ValueError("Approved Research v2 selection changed; create a new release")
    selection = read(path.name)
    for name, expected in selection["source"]["local_evidence_file_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned Research evidence changed: " + name)
    return build(selection, read("cost-evidence-coverage-v1.json"), hour_release.build_current())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Publish generated metadata and archive hour-v1")
    args = parser.parse_args()
    value = build_current()
    if args.write:
        active = ROOT / "release.json"
        old = hour_release.build_current()
        current = json.loads(active.read_text())
        if current not in (old, value):
            raise ValueError("Refusing to replace another release")
        archive = ROOT / "release-hour-v1.json"
        if archive.exists() and json.loads(archive.read_text()) != old:
            raise ValueError("Historical hour-v1 archive differs")
        if not archive.exists():
            if current != old:
                raise ValueError("Missing historical hour-v1 archive")
            archive.write_bytes(active.read_bytes())
        active.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"release_id": value["release_id"], "case_count": value["case_count"],
                      "files": value["files"], "sha256": public.digest(value),
                      "backend_counts": value["backend_counts"]}))


if __name__ == "__main__":
    main()
