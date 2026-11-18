#!/usr/bin/env python3
"""Trusted scorer. Reads sealed predictions + private responses; never executes submission code.

Writes to --output:
  reward.json          {"score": V3 task score, "reward": V3 task score,
                        "v3_<ctx>": V3 context scores, "v2_<ctx>": v2 WMSE-only skills,
                        "v1_*": historical cell-eval panel numbers}
  report.json          full per-context V3 dual-skill measurement, query-independence audit,
                       v2 WMSE measurement, and v1 per-subset / per-stratum report (operator-facing)
  agent_feedback.json  v1 leaderboard-subset scores only (what the runner may return to the agent)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from calibrated_reward_v2 import (
    V3_VERSION,
    VERSION as V2_VERSION,
    aggregate_contexts,
    aggregate_contexts_v3,
)
from cellscore import cell_eval_version, score_context, sha256
from population_metrics_v2 import (
    bundle_repeat_nir,
    load_or_build_bundle,
    prediction_means,
    score_against_bundle,
)
from query_independence import DEFAULT_ATOL, DEFAULT_RTOL, compare_prediction_files


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--benchmark", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--private", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--wait-seconds", type=int, default=14400)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--expected-contexts", nargs="+")
    ap.add_argument("--require-query-independence", action="store_true")
    ap.add_argument("--query-independence-atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--query-independence-rtol", type=float, default=DEFAULT_RTOL)
    args = ap.parse_args()
    ready = args.sealed.parent / "READY"
    deadline = time.monotonic() + args.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction snapshot was not published")
        time.sleep(2)
    snapshot_path = args.sealed / "snapshot.json"
    if ready.read_text().strip() != sha256(snapshot_path):
        raise ValueError("snapshot ready hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema") not in ("virtual-cell-zeroshot-snapshot/v1", "virtual-cell-zeroshot-snapshot/v2"):
        raise ValueError(f"Unsupported snapshot schema: {snapshot.get('schema')}")
    enforce_qi = (
        args.require_query_independence
        or snapshot.get("schema") == "virtual-cell-zeroshot-snapshot/v2"
        or any(
            isinstance(v, dict) and ("predictions_subset_sha256" in v or "subset_failed" in v)
            for v in snapshot.get("contexts", {}).values()
        )
    )
    contexts = sorted(p.name for p in args.private.iterdir() if (p / "responses.h5ad").exists())
    public_contexts = {p.name for p in args.benchmark.iterdir() if (p / "public/requests.jsonl").is_file()}
    expected = set(args.expected_contexts) if args.expected_contexts else public_contexts
    if not expected or set(contexts) != expected or public_contexts != expected:
        raise ValueError("Benchmark inputs and private responses must cover exactly the expected contexts")
    report = {
        "schema": "virtual-cell-zeroshot-report/v3",
        "status": "ok",
        "cell_eval_version": cell_eval_version(),
        "reward_version": V3_VERSION,
        "v2_reward_version": V2_VERSION,
        "evaluation_mode": "isolated_two_pass_model_replay" if enforce_qi else "isolated_model_replay",
        "contexts": {},
    }
    finals, boards = [], []
    v2_inputs, v3_inputs, v2_details, tiers = {}, {}, {}, {}
    qi_by_ctx: dict[str, dict] = {}
    scratch = args.output / "scratch"
    for ctx in contexts:
        priv = args.private / ctx
        pm = json.loads((priv / "manifest.json").read_text())
        if pm["responses_sha256"] != sha256(priv / "responses.h5ad"):
            raise ValueError(f"{ctx}: private response checksum mismatch")
        tiers[ctx] = str(pm["tier"])
        public = args.benchmark / ctx / "public"
        entry = snapshot["contexts"].get(ctx)
        if pm["requests_sha256"] != sha256(public / "requests.jsonl") or pm["public_basal_sha256"] != sha256(public / "basal.h5ad"):
            raise ValueError(f"{ctx}: public inputs differ from the private manifest")
        bundle = load_or_build_bundle(priv, public / "requests.jsonl")
        anchors = {k: bundle[k] for k in ("negative_loss", "repeat_loss", "gap_lower95")}
        rep_nir = bundle_repeat_nir(bundle)
        n_targets = len(bundle["roster"])
        v2_details[ctx] = {
            "n_targets": n_targets,
            "excluded_targets": bundle["excluded_targets"],
            "gap": bundle["gap"],
            "gap_upper95": bundle["gap_upper95"],
            "bundle": bundle["cache"],
            "nir": None,
            "negative_nir": 0.5,
            **rep_nir,
        }
        if not entry or entry.get("failed"):
            report["contexts"][ctx] = {
                "status": "failed",
                "error": (entry or {}).get("error", "missing"),
                "tier": pm["tier"],
                "v1_final_score": 0.0,
                "v1_leaderboard_score": 0.0,
            }
            finals.append(0.0)
            boards.append(0.0)
            v2_inputs[ctx] = {"loss": None, **anchors, "status": "invalid_submission"}
            v3_inputs[ctx] = {
                "loss": None,
                **anchors,
                "nir": None,
                "repeat_nir": rep_nir["repeat_nir"],
                "targets": n_targets,
                "status": "invalid_submission",
            }
            qi_by_ctx[ctx] = {"status": "context_failed", "passed": True, "changed_requests": 0}
            continue
        pred = args.sealed / ctx / "predictions.h5ad"
        if (
            entry["predictions_sha256"] != sha256(pred)
            or entry["requests_sha256"] != sha256(public / "requests.jsonl")
            or entry["basal_sha256"] != sha256(public / "basal.h5ad")
        ):
            raise ValueError(f"{ctx}: sealed snapshot provenance mismatch")

        # Two-pass per-request query-independence check
        if enforce_qi:
            if entry.get("subset_failed"):
                qi_by_ctx[ctx] = {
                    "status": "failed",
                    "passed": False,
                    "reason": entry.get("subset_error", "subset inference failed"),
                    "changed_requests": -1,
                }
            elif "predictions_subset_sha256" not in entry or "requests_subset_sha256" not in entry:
                qi_by_ctx[ctx] = {
                    "status": "failed",
                    "passed": False,
                    "reason": "missing sealed subset predictions for two-pass query independence check",
                    "changed_requests": -1,
                }
            else:
                pred_sub = args.sealed / ctx / "predictions_subset.h5ad"
                req_sub = args.sealed / ctx / "requests_subset.jsonl"
                if (
                    not pred_sub.is_file()
                    or not req_sub.is_file()
                    or entry["predictions_subset_sha256"] != sha256(pred_sub)
                    or entry["requests_subset_sha256"] != sha256(req_sub)
                ):
                    raise ValueError(f"{ctx}: sealed subset snapshot provenance mismatch")
                try:
                    qi_res = compare_prediction_files(
                        pred,
                        pred_sub,
                        public / "basal.h5ad",
                        public / "requests.jsonl",
                        req_sub,
                        atol=args.query_independence_atol,
                        rtol=args.query_independence_rtol,
                    )
                    offending = {
                        rid: info["max_absolute_logmean_difference"]
                        for rid, info in qi_res["requests"].items()
                        if not info["within_numerical_tolerance"]
                    }
                    qi_by_ctx[ctx] = {
                        "status": qi_res["status"],
                        "passed": qi_res["passed"],
                        "atol": qi_res["atol"],
                        "rtol": qi_res["rtol"],
                        "checked_requests": qi_res["checked_requests"],
                        "changed_requests": qi_res["changed_requests"],
                        "max_difference": qi_res["max_difference"],
                        "offending_requests": offending,
                        "subset_seconds": entry.get("subset_seconds"),
                    }
                except ValueError as qi_err:
                    qi_by_ctx[ctx] = {
                        "status": "failed",
                        "passed": False,
                        "reason": str(qi_err),
                        "changed_requests": -1,
                    }

        measured = score_against_bundle(
            prediction_means(pred, public / "basal.h5ad", public / "requests.jsonl", bundle["roster"]),
            bundle,
        )
        v2_inputs[ctx] = {"loss": measured["loss"], **anchors}
        v3_inputs[ctx] = {
            "loss": measured["loss"],
            **anchors,
            "nir": measured["nir"],
            "repeat_nir": rep_nir["repeat_nir"],
            "targets": n_targets,
        }
        v2_details[ctx].update(
            per_target_loss=measured["per_target_loss"],
            nir=measured["nir"],
            per_target_nir=measured["per_target_nir"],
        )
        subsets = {
            "leaderboard": [r["request_id"] for r in pm["requests"] if r["subset"] == "leaderboard"],
            "final": [r["request_id"] for r in pm["requests"] if r["subset"] == "final"],
        }
        groups = {
            "stratum": {r["request_id"]: r["stratum"] for r in pm["requests"]},
            "target_seen": {
                r["request_id"]: ("withheld" if r["withheld"] else ("seen" if r["seen_in_training"] else "unseen"))
                for r in pm["requests"]
            },
            "final_by_stratum": {r["request_id"]: r["stratum"] for r in pm["requests"] if r["subset"] == "final"},
        }
        baseline_dir = priv / "noeffect" if (priv / "noeffect").exists() else None
        res = score_context(
            pred,
            priv / "basal.h5ad",
            priv / "responses.h5ad",
            public / "requests.jsonl",
            scratch / ctx,
            subsets=subsets,
            groups=groups,
            threads=args.threads,
            baseline_dir=baseline_dir,
        )
        final = res["subsets"]["final"]["score"]
        board = res["subsets"]["leaderboard"]["score"]
        report["contexts"][ctx] = {
            "status": "ok",
            "tier": pm["tier"],
            "chemistry": pm["chemistry"],
            "v1_final_score": final,
            "v1_leaderboard_score": board,
            "v1_all_score": res["all"]["score"],
            "v1_final": res["subsets"]["final"],
            "v1_leaderboard": res["subsets"]["leaderboard"],
            "v1_all": res["all"],
            "v1_groups": res["groups"],
            "inference_seconds": entry.get("seconds"),
        }
        finals.append(final)
        boards.append(board)

    v2 = aggregate_contexts(v2_inputs, expected_contexts=contexts)
    v3 = aggregate_contexts_v3(v3_inputs, expected_contexts=contexts)
    unique_tiers = len(set(tiers.values())) == len(contexts)
    v3_contexts_by_tier = {}
    for ctx in contexts:
        report["contexts"][ctx]["v2"] = {**v2["contexts"][ctx], **v2_details[ctx]}
        report["contexts"][ctx]["v3"] = v3["contexts"][ctx]
        if ctx in qi_by_ctx:
            report["contexts"][ctx]["query_independence"] = qi_by_ctx[ctx]
        tier_key = tiers[ctx] if unique_tiers else ctx
        v3_contexts_by_tier[tier_key] = v3["contexts"][ctx]

    report["v3_contexts"] = v3_contexts_by_tier
    report["v3_aggregation"] = v3["aggregation"]
    report["v2_reward"] = v2["reward"]
    report["v2_aggregation"] = v2["aggregation"]
    report["v1_reward"] = float(np.mean(finals)) if finals else 0.0
    report["v1_leaderboard_reward"] = float(np.mean(boards)) if boards else 0.0

    failed_qi_contexts = sorted(c for c, q in qi_by_ctx.items() if not q.get("passed", True))
    if enforce_qi:
        report["query_independence"] = {
            "enforced": True,
            "status": "failed" if failed_qi_contexts else "passed",
            "passed": not failed_qi_contexts,
            "atol": args.query_independence_atol,
            "rtol": args.query_independence_rtol,
            "failed_contexts": failed_qi_contexts,
            "total_changed_requests": sum(
                max(0, int(q.get("changed_requests", 0))) for q in qi_by_ctx.values()
            ),
        }
    else:
        report["query_independence"] = {"enforced": False, "status": "not_provided", "passed": True}

    if enforce_qi and failed_qi_contexts:
        report["status"] = "rejected_query_dependence"
        report["unpenalized_v3_score"] = v3["score"]
        report["score"] = 0.0
        report["reward"] = 0.0
        reward_payload = {
            "score": 0.0,
            "reward": 0.0,
            "query_independence_failed": 1.0,
        }
    else:
        report["score"] = v3["score"]
        report["reward"] = v3["score"]
        reward_payload = {
            "score": report["score"],
            "reward": report["reward"],
            **{f"v3_{c}": v["v3"]["context_score"] for c, v in report["contexts"].items()},
            **{
                f"v3_{c}_signed": v["v3"]["signed_context"]
                for c, v in report["contexts"].items()
                if v["v3"]["signed_context"] is not None
            },
            **{
                f"v3_{c}_amplitude_skill": v["v3"]["amplitude_skill"]
                for c, v in report["contexts"].items()
                if v["v3"]["amplitude_skill"] is not None
            },
            **{
                f"v3_{c}_identification_skill": v["v3"]["identification_skill"]
                for c, v in report["contexts"].items()
                if v["v3"]["identification_skill"] is not None
            },
            "v2_reward": report["v2_reward"],
            **{f"v2_{c}": v["v2"]["reward"] for c, v in report["contexts"].items()},
            **{
                f"v2_{c}_raw_skill": v["v2"]["raw_skill"]
                for c, v in report["contexts"].items()
                if v["v2"]["raw_skill"] is not None
            },
            **{
                f"v2_{c}_nir": v["v2"]["nir"]
                for c, v in report["contexts"].items()
                if v["v2"].get("nir") is not None
            },
            **{
                f"v2_{c}_repeat_nir": v["v2"]["repeat_nir"]
                for c, v in report["contexts"].items()
                if v["v2"].get("repeat_nir") is not None
            },
            "v1_reward": report["v1_reward"],
            "v1_leaderboard_reward": report["v1_leaderboard_reward"],
            **{f"v1_{c}_final": v["v1_final_score"] for c, v in report["contexts"].items()},
            **{f"v1_{c}_leaderboard": v["v1_leaderboard_score"] for c, v in report["contexts"].items()},
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "agent_feedback.json").write_text(json.dumps({
        "schema": "virtual-cell-zeroshot-feedback/v2",
        "cell_eval_version": report["cell_eval_version"],
        "v1_leaderboard_reward": report["v1_leaderboard_reward"],
        "contexts": {
            c: {
                "status": v["status"],
                "tier": v["tier"],
                "v1_leaderboard_score": v["v1_leaderboard_score"],
                "v1_leaderboard": v.get("v1_leaderboard"),
                "error": v.get("error"),
            }
            for c, v in report["contexts"].items()
        },
    }, indent=2, sort_keys=True) + "\n")
    (args.output / "reward.json").write_text(json.dumps(harbor_reward(reward_payload), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
