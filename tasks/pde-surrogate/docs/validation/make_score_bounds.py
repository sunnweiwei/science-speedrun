#!/usr/bin/env python3
"""Derive the track's score bounds from the validation records, so the two can never drift.

    python3 make_score_bounds.py            # rewrites docs/validation/score_bounds.json and environment/evaluation/score_bounds.json
    python3 make_score_bounds.py --check    # exit 1 if either file differs from what the records imply

Lower bound: persistence (repeat the last context frame), a real predictor every submission
should beat. Upper bound: the per-group oracle, the best group VRMSE any measured package reached,
which routes each group to a different model and is therefore not achievable by one legal
package; it is a ceiling on the measured population, and the gap to it is the task's headroom.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
from score import reward_from_vrmse  # noqa: E402

GROUPS = ("ape2d", "well/turbulent_radiative_layer_2D", "well/active_matter", "pdearena/navier_stokes_2d")
# (record, label, legal single package?) - every measured package on the canonical split
PACKAGES = [
    ("pdet_zero_shot_mc-s_canonical.json", "PDE-Transformer mc-s, zero-shot", True),
    ("pdet_zero_shot_mc-b_canonical.json", "PDE-Transformer mc-b, zero-shot", True),
    ("pdet_zero_shot_mc-l_canonical.json", "PDE-Transformer mc-l, zero-shot", True),
    ("pdet_zero_shot_mc-s_transfer.json", "PDE-Transformer mc-s, zero-shot", True),
    ("pdet_zero_shot_mc-b_transfer.json", "PDE-Transformer mc-b, zero-shot", True),
    ("pdet_zero_shot_mc-l_transfer.json", "PDE-Transformer mc-l, zero-shot", True),
    ("dpot_ti_zero_shot_canonical.json", "DPOT-Ti, zero-shot", True),
    ("dpot_m_zero_shot_canonical.json", "DPOT-M, zero-shot", True),
    ("dpot_ti_zero_shot_transfer.json", "DPOT-Ti, zero-shot", True),
    ("dpot_m_zero_shot_transfer.json", "DPOT-M, zero-shot", True),
    ("task_trained_baseline.json", "task starter (from scratch, 40 min)", True),
    ("well_cnext_trl2d_from_scratch.json", "CNextU-net from scratch, turbulent layer", True),
    ("well_cnext_active_matter_from_scratch.json", "CNextU-net from scratch, active matter", True),
    ("well_cnext_pdearena_from_scratch.json", "CNextU-net from scratch, PDEArena", True),
    ("pdet_finetune_trl2d.json", "PDE-Transformer sc-s fine-tuned, turbulent layer", True),
    ("pdet_finetune_active_matter.json", "PDE-Transformer sc-s fine-tuned, active matter", True),
    ("pdet_finetune_pdearena.json", "PDE-Transformer sc-s fine-tuned, PDEArena", True),
    ("pdet_finetune_joint.json", "PDE-Transformer sc-s fine-tuned on all three transfer pools", True),
]


def groups_of(doc):
    """{group: vrmse} and the whole-package reward, from either record schema."""
    res = doc.get("result", doc)
    g = res["groups"]
    return {k: (v["vrmse"] if isinstance(v, dict) else v) for k, v in g.items()}, res.get("reward")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    trivial = {r["predictor"]: r for r in json.loads((HERE / "trivial_predictors_canonical.json").read_text())["results"]}
    persist, mean, zero = trivial["persist"], trivial["mean"], trivial["zero"]
    lower = persist["reward"]

    best = {g: None for g in GROUPS}
    measured = {}          # label -> whole-package reward, for packages that cover all four groups
    for name, label, legal in PACKAGES:
        doc = json.loads((HERE / name).read_text())
        g, reward = groups_of(doc)
        for grp, v in g.items():
            if grp in best and (best[grp] is None or v < best[grp]["vrmse"]):
                best[grp] = {"vrmse": round(v, 4), "reward": reward_from_vrmse(v), "model": label, "source": f"docs/validation/{name}"}
        if set(g) == set(GROUPS) and legal:
            measured[label] = reward
    # released checkpoints replayed zero-shot: the corpus group from the canonical record, the three transfer
    # groups from the transfer record; their whole-package reward is the mean of the four group rewards
    for canonical, transfer, label in (("pdet_zero_shot_mc-s", "pdet_zero_shot_mc-s", "PDE-Transformer mc-s, zero-shot"),
                                       ("pdet_zero_shot_mc-b", "pdet_zero_shot_mc-b", "PDE-Transformer mc-b, zero-shot"),
                                       ("pdet_zero_shot_mc-l", "pdet_zero_shot_mc-l", "PDE-Transformer mc-l, zero-shot"),
                                       ("dpot_ti_zero_shot", "dpot_ti_zero_shot", "DPOT-Ti, zero-shot"),
                                       ("dpot_m_zero_shot", "dpot_m_zero_shot", "DPOT-M, zero-shot")):
        g = {}
        for part in (f"{canonical}_canonical.json", f"{transfer}_transfer.json"):
            g.update(groups_of(json.loads((HERE / part).read_text()))[0])
        assert set(g) == set(GROUPS), (label, sorted(g))
        measured[label] = sum(reward_from_vrmse(g[grp]) for grp in GROUPS) / len(GROUPS)
    upper = sum(b["reward"] for b in best.values()) / len(GROUPS)
    best_legal = max(measured.items(), key=lambda kv: kv[1])
    joint = json.loads((HERE / "pdet_finetune_joint.json").read_text())["result"]["reward"]
    norm = lambda r: (r - lower) / (upper - lower)

    table = {"zero prediction": zero["reward"], "persistence": lower, "spatial mean of last frame": mean["reward"],
             **{k: v for k, v in measured.items()}}
    n_cases = sum(1 for c in json.loads((ROOT / "environment" / "data" / "split.json").read_text())["cases"] if c["split"] == "test")
    record = {
        "what": "reference lower and upper bound for the pde-surrogate track, and the 0-1 normalisation they define",
        "generated_by": "docs/validation/make_score_bounds.py from the records it names; do not edit by hand",
        "protocol": f"task protocol: canonical split (environment/data/split.json), all {n_cases:,} held-out cases. Every input "
                    "number is a group VRMSE already measured and recorded in docs/validation/; nothing here required a new run.",
        "why": f"the raw contract reward has almost no dynamic range. A package that predicts nothing scores {zero['reward']:.4f} "
               f"and the best legal package measured scores {best_legal[1]:.4f}, so the whole meaningful band is "
               f"{best_legal[1] - zero['reward']:.2f} wide on a 0-1 scale. Normalising against a real floor and a real ceiling "
               "makes the number readable and comparable with other tracks.",
        "reward_definition": "mean over the four groups of (1 - log10(clamp(group VRMSE, 0.01, 10))) / 3; see environment/evaluation/score.py",
        "lower_bound": {"reward": lower, "what": "persistence: repeat the last context frame for every step",
                        "rationale": f"a real predictor rather than a degenerate one. Zero prediction scores lower ({zero['reward']:.4f}) "
                                     "but is not a baseline any submission would fall below by accident.",
                        "per_group_vrmse": {g: round(persist["groups"][g], 4) for g in GROUPS},
                        "source": "docs/validation/trivial_predictors_canonical.json"},
        "upper_bound": {"reward": upper, "what": "per-group oracle: the best VRMSE any measured model reached on each group",
                        "achievable_by_a_submission": False,
                        "caveat": "this routes each group to a different model, which environment/contract.md forbids for a submission "
                                  "(checkpoints selected at inference time by dataset name). It is a ceiling on what the measured "
                                  f"population can do, not a target a legal package has reached. The best legal single package measured is "
                                  f"{best_legal[0]} at {best_legal[1]:.4f}, which normalises to {norm(best_legal[1]):.3f} - the track has no "
                                  "strong legal solution yet, and that gap is the headroom an agent is being offered.",
                        "per_group": best,
                        "attempt_to_replace_with_a_legal_package": (
                            "one sc-s fine-tuned on all three transfer pools at once was trained and replayed for exactly this purpose "
                            f"(docs/validation/pdet_finetune_joint.json). It scored {joint:.4f}, below the persistence floor, and is worse than "
                            "the per-dataset fine-tunes on every group (negative transfer). The 128x128 crop upstream's loader forces was "
                            "checked as an explanation and ruled out: the one group that was never cropped, and that had more training "
                            "passes than its dedicated run, degraded just as much. The oracle bound therefore stands, and closing the gap "
                            "to it looks like a research problem rather than a matter of running the obvious baseline longer.")},
        "normalisation": "score = clip((reward - lower) / (upper - lower), 0, 1)",
        "normalised_measurements": {k: {"reward": round(v, 4), "normalised": round(norm(v), 4)}
                                    for k, v in sorted(table.items(), key=lambda kv: kv[1])},
    }
    evaluation = {
        "schema": "benchmark-score-bounds/v1", "track": "pde-surrogate",
        "metric": "reward, the unweighted mean over the four groups of a log-band map of each group's mean VRMSE (score.py)",
        "normalisation": "score = clip((reward - lower.reward) / (upper.reward - lower.reward), 0, 1)",
        "lower": {"reward": lower, "what": record["lower_bound"]["what"], "rationale": record["lower_bound"]["rationale"],
                  "source": "docs/validation/trivial_predictors_canonical.json"},
        "upper": {"reward": upper, "what": record["upper_bound"]["what"], "achievable_by_a_submission": False,
                  "caveat": "this routes each group to a different model, which environment/contract.md forbids for a submission. "
                            f"The best legal single package measured is {best_legal[1]:.4f}, which normalises to {norm(best_legal[1]):.3f}. "
                            "A joint-pool fine-tune was trained specifically to replace this bound with a legal package and came out "
                            "below the floor (docs/validation/pdet_finetune_joint.json), so the gap is a research gap rather than an unrun baseline.",
                  "source": "docs/validation/score_bounds.json"},
        "note": f"the raw reward has almost no dynamic range: a package that predicts nothing scores {zero['reward']:.4f} and the best "
                f"legal package {best_legal[1]:.4f}, so the whole meaningful band is {best_legal[1] - zero['reward']:.2f} wide. The normalised "
                "score is what should be compared across tracks; reward.txt keeps the raw value.",
    }
    outputs = {HERE / "score_bounds.json": record, ROOT / "environment" / "evaluation" / "score_bounds.json": evaluation}
    changed = []
    for path, doc in outputs.items():
        text = json.dumps(doc, indent=1) + "\n"
        if not path.exists() or path.read_text() != text:
            changed.append(path.name)
            if not a.check:
                path.write_text(text)
    print(f"lower {lower:.4f} (persistence)  upper {upper:.4f} (oracle)  best legal {best_legal[0]} {best_legal[1]:.4f} -> {norm(best_legal[1]):.3f}")
    for g, b in best.items():
        print(f"  {g:36s} {b['vrmse']:.4f}  {b['model']}")
    if a.check and changed:
        print("out of date:", changed); sys.exit(1)
    print("written:" if changed else "unchanged:", changed or list(p.name for p in outputs))


if __name__ == "__main__":
    main()
