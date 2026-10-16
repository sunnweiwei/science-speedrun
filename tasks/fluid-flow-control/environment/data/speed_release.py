"""Freeze a measured-cost selection, without asserting CFD protocol certification."""
import argparse
import json
import math
from pathlib import Path

import public_release as public

ROOT = Path(__file__).resolve().parent
RELEASE_ID = "hydrogym-public-speed-v1"


def read(name):
    return json.loads((ROOT / name).read_text())


def build(selection, evidence):
    if selection["release_id"] != RELEASE_ID or selection["status"] != "frozen":
        raise ValueError("Unknown/unfrozen speed selection")
    if selection["max_source_bytes"] != 500_000_000_000:
        raise ValueError("Wrong speed-selection source budget")
    if (selection['policy']['measured_control_seconds_upper_bound']!=30
            or selection['policy']['minimum_post_warmup_samples']!=2):
        raise ValueError("Wrong frozen speed-screening rule")
    ids = selection["case_ids"]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Expected a nonempty sorted unique case list")
    if selection["cost_evidence_sha256"] != public.digest(evidence):
        raise ValueError("Speed evidence has changed")
    if evidence['source_revision'] != public.REVISION or evidence['code_revision'] != public.CODE_REVISION:
        raise ValueError("Speed evidence source/code revision mismatch")
    records = {r["case"]: r for r in evidence["cases"]}
    if len(records) != len(evidence["cases"]):
        raise ValueError("Duplicate speed evidence")
    for name in ids:
        r = records.get(name, {})
        seconds = r.get("seconds_per_control")
        if (r.get("status") != "complete" or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0 or seconds > 30
                or r.get("measured_controls", 0) < 2):
            raise ValueError(f"No completed acceptable short timing for {name}")
        reset = r.get('reset_seconds')
        if (type(reset) not in (int, float) or not math.isfinite(reset)
                or reset <= 0 or r.get('container_exit_code') != 0):
            raise ValueError(f"No clean completed reset/control/reset probe for {name}")
        samples=r.get('timing_samples_seconds',[])
        if (len(samples)!=r['measured_controls'] or any(type(x) not in (int,float)
                or not math.isfinite(x) or x<=0 for x in samples)
                or not math.isclose(sum(samples)/len(samples),seconds,rel_tol=1e-10)):
            raise ValueError(f"Timing summary/sample mismatch for {name}")
    result = public.build(read("resource-audit.json"), read("source-files.json"),
                          read("native-configs.json"), selected_case_ids=ids)
    if not any(c["dimension"] == 3 for c in result["cases"]):
        raise ValueError("The benchmark must retain 3D control")
    if any(c["backend"] not in ("maia", "jax", "jax-fluids") for c in result["cases"]):
        raise ValueError("Unreviewed backend in this speed-first selection")
    for c in result['cases']:
        r=records[c['id']]
        if (r['backend']!=c['backend'] or r['dimension']!=c['dimension']
                or r['data_bytes']!=c['bytes'] or r['hardware']!='1 H100'):
            raise ValueError("Timing evidence belongs to a different case/backend/resource")
    excluded = selection["excluded_complete_resources"]
    by_exclusion = {r["case"]: r for r in excluded}
    if len(by_exclusion) != len(excluded) or set(by_exclusion) & set(ids):
        raise ValueError("Ambiguous selection/exclusion")
    actual_excluded = {r["environment"] for r in result["exclusions"]
                       if r["reason"] == "not_selected_500gb_scope"}
    if set(by_exclusion) != actual_excluded:
        raise ValueError("Every unselected complete resource needs an explicit reason")
    if selection['candidate_inventory_count'] != len(ids)+len(actual_excluded):
        raise ValueError("Incomplete candidate inventory accounting")
    for r in result["exclusions"]:
        if r["reason"] == "not_selected_500gb_scope":
            decision = by_exclusion[r["environment"]]
            r.update(reason=decision["reason"], detail=decision["detail"])
    result.update(release_id=RELEASE_ID, created="2026-09-07",
        supersedes=public.RELEASE_ID,
        scope="Cost-aware known-task selection from all 62 resource-complete public variants; retain diverse 3D control, not the original paper's full benchmark.",
        selection_spec_sha256=public.digest(selection),
        cost_evidence_sha256=public.digest(evidence),
        selection_policy=selection["policy"],
        reconstruction_policy="No source geometry/restart/mesh/time step is modified. Short-loop timing is not full-episode physical or evaluator certification; any later membership or physical protocol change is separately versioned.")
    for case in result["cases"]:
        case["cost_screening"] = records[case["id"]]
        case["execution_status"] = "short_native_loop_passed_full_evaluation_pending"
        case['cost_configuration_basis'] = (
            'Pinned ChannelFlowSpectralEnv class defaults (72^3, 24 actions, 50 CFD substeps/action), not conflicting asset YAML; bind explicitly in the reviewed evaluation protocol.'
            if case['id']=='Channel_3D_Retau180' else
            'Unmodified source-native YAML control interval, action bounds, reward and mesh; final sensor/horizon/replay certification remains required.')
    return result


def build_current():
    return build(read("selection-speed-v1.json"), read("cost-evidence-speed-v1.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Archive the original release and write the newly generated default")
    args = parser.parse_args()
    value = build_current()
    if args.write:
        legacy = public.build(read("resource-audit.json"), read("source-files.json"), read("native-configs.json"))
        archive = ROOT / "release-500gb-v1.json"
        if archive.exists() and json.loads(archive.read_text()) != legacy:
            raise ValueError("Refusing to change an existing historical manifest")
        current = read("release.json")
        if current not in (legacy, value):
            raise ValueError("Refusing to overwrite an unrelated release")
        if not archive.exists():
            archive.write_text(json.dumps(legacy, indent=2) + "\n")
        (ROOT / "release.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"release_id": value["release_id"], "case_count": value["case_count"],
                      "files": value["files"], "sha256": public.digest(value)}))


if __name__ == "__main__":
    main()
