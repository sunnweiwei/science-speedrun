"""Independent closed-loop replay of a policy package across the frozen case suite.

Usage (inside the declared runtime):
  python evaluation/harbor_verify.py --submission /workspace/submission --output OUT \
      [--cases NAME ...] [--seeds 0 1 2] [--randomization visible|none|FILE] [--workers N]

The evaluator owns the simulator, the perturbation distribution and the metrics; the
policy only supplies coil voltages. Outputs: reward.json (flat numeric, `reward` = success
fraction), results.json (complete records), grading.json (scope), per-episode logs.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tokagym import CASES, ENV_VERSION, TokamakMagneticEnv  # noqa: E402
from tokagym import metrics as M  # noqa: E402
from tokagym.env import VISIBLE_RANDOMIZATION  # noqa: E402
import protocol  # noqa: E402
import supervise  # noqa: E402

METRIC_VERSION = "tokamak-shape-metrics/v3"
DEFAULT_SEEDS = [0, 1, 2]
VOID_LIMIT = 0.2   # more void episodes than this is an infrastructure failure, not a score


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def load_randomization(spec):
    if spec in ("visible", None):
        return dict(VISIBLE_RANDOMIZATION), "visible", DEFAULT_SEEDS
    if spec == "none":
        return "none", "none", DEFAULT_SEEDS
    data = json.loads(Path(spec).read_text())
    return data["randomization"], data.get("name", Path(spec).stem), data.get("seeds", DEFAULT_SEEDS)


# up to 4.5 s per step for the 1200-step sequence case; a stalled solver is caught by the silence limit
EPISODE_TIMEOUT_SEC = float(os.environ.get("TOKAGYM_EPISODE_TIMEOUT", "5400"))
STEP_SILENCE_SEC = float(os.environ.get("TOKAGYM_STEP_SILENCE", "600"))   # no completed step for this long -> kill
# Real-time gate: the policy's own computation per step must fit in one control period (Degrave et al.
# ran in 50 us). It is the median over steps of (submission round trip - null-policy round trip), the
# null policy answering the same observation immediately after the submission on every step.
LATENCY_LIMIT_MS = 1e3 * 5e-4
NULL_POLICY = HERE / "null_policy"


def open_null_policy(case_name, env, out, tag, timeout):
    """The null policy, reset for this case; timed on every step next to the submission."""
    null = protocol.Policy(NULL_POLICY, out / f"null-{tag}.stderr", timeout=timeout)
    null.reset({"environment": case_name, "family": CASES[case_name].family, "seed": -1},
               [env.n_obs], [env.n_active], (-env.vmax).tolist(), env.vmax.tolist())
    return null


def run_case_stream(args, seeds, queue):
    """Child process: replay the given seeds of one case, streaming one message per episode."""
    case_name, submission, out_dir, randomization, timeout = args
    out = Path(out_dir) / case_name
    out.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(s) for s in seeds)
    policy = null = None
    try:
        env = TokamakMagneticEnv(case_name, randomization=randomization, solver="nonlinear",
                                 strict_actions=True)
        policy = protocol.Policy(submission, out / f"policy-{tag}.stderr", timeout=timeout)
        low = (-env.vmax).tolist()
        high = env.vmax.tolist()
        for seed in seeds:
            started = time.monotonic()
            obs, info = env.reset(seed=seed)
            policy.reset({"environment": case_name, "family": CASES[case_name].family,
                          "seed": seed, "machine": env.spec.name, "env_version": ENV_VERSION,
                          "control_dt": 5e-4, "horizon": env.horizon},
                         [env.n_obs], [env.n_active], low, high)
            n0 = len(policy.latencies)
            # Real-time gate: the null policy answers the same observation next to the submission on
            # every step, so both round trips see the same machine load. The order alternates: the
            # first process to answer after a solver step pays the wake-up/cold-cache cost, which a
            # fixed order would charge to the submission alone.
            null = open_null_policy(case_name, env, out, tag, timeout)
            m0 = len(null.latencies)
            ret = 0.0
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                packed = protocol.unpack(obs)
                if step % 2 == 0:
                    action = policy.act(packed)
                try:
                    null.act(packed)
                except protocol.PolicyError as error:   # our reference process, never the submission's fault
                    raise protocol.InfrastructureError(f"null policy failed: {error}") from error
                if step % 2 == 1:
                    action = policy.act(packed)
                step += 1
                obs, reward, terminated, truncated, step_info = env.step(action)
                ret += reward
                queue.put({"kind": "heartbeat", "shape_rms": step_info["shape_rms"],
                           "ip_error_rel": step_info["ip_error_rel"]})
            summary = env.episode_summary()
            lat = np.array(policy.latencies[n0:])
            null_lat = np.array(null.latencies[m0:])
            null.close()
            null = None
            n = min(len(lat), len(null_lat))
            null_ms = float(np.median(null_lat) * 1e3) if len(null_lat) else None
            summary.update({
                "kind": "episode", "seed": seed, "return_diagnostic": ret,
                "policy_latency_ms_median": float(np.median(lat) * 1e3) if len(lat) else None,
                "policy_latency_ms_p95": float(np.percentile(lat, 95) * 1e3) if len(lat) else None,
                "null_latency_ms_median": null_ms,
                "policy_compute_ms": (max(0.0, float(np.median(lat[:n] - null_lat[:n]) * 1e3)) if n else None),
                "wall_time_sec": time.monotonic() - started,
                "setup": {k: info[k] for k in ("plasma_resistivity", "passive_scale", "supply_delay_steps",
                                               "reset_attempts", "reset_fallback", "profile_constraint",
                                               "drift", "crashes", "dead_channels", "handover_steps")},
            })
            (out / f"episode-{seed}.json").write_text(json.dumps(summary, indent=1))
            queue.put(summary)
        policy.close()
        policy = None
        queue.put({"kind": "case", "status": "ok", "error": None})
    except protocol.PolicyError as error:
        queue.put({"kind": "case", "status": "policy_error", "error": str(error)})
    except protocol.InfrastructureError as error:
        queue.put({"kind": "case", "status": "infrastructure_error", "error": str(error)})
    except Exception:  # noqa: BLE001
        queue.put({"kind": "case", "status": "infrastructure_error", "error": traceback.format_exc()[-2000:]})
    finally:
        if policy is not None:
            policy.abort()
        if null is not None:
            null.abort()


def run_all(jobs, seeds, workers):
    """Supervise one process per (case, seed); a hung episode is killed and recorded as solver_timeout.

    Episodes of one case run in parallel (the long sequence case would otherwise dominate the
    wall time); results are merged per case, keeping the worst status.
    """
    episode_jobs = [job for job in jobs for _ in seeds]
    episode_seeds = [[seed] for _ in jobs for seed in seeds]
    raw = supervise.run_supervised(run_case_stream, episode_jobs, episode_seeds, EPISODE_TIMEOUT_SEC,
                                   workers, silence_timeout=STEP_SILENCE_SEC)
    rank = {"ok": 0, "policy_error": 1, "infrastructure_error": 2}
    merged = {}
    for i, job in enumerate(episode_jobs):
        r = raw[i]
        entry = merged.setdefault(job[0], {"case": job[0], "family": CASES[job[0]].family, "episodes": [],
                                           "status": "ok", "error": None, "supervisor_notes": []})
        entry["episodes"] += [{k: v for k, v in e.items() if k != "kind"} for e in r["episodes"]]
        status, error = "ok", None
        if r["case"] is not None:
            status, error = r["case"]["status"], r["case"]["error"]
        if r["errors"]:
            crash = [e for e in r["errors"] if "exceeded" not in e and "silent" not in e]
            if crash and status == "ok":
                status, error = "infrastructure_error", "; ".join(crash)[:2000]
        entry["supervisor_notes"] += [e for e in r["errors"] if "exceeded" in e or "silent" in e]
        if rank[status] > rank[entry["status"]]:
            entry["status"], entry["error"] = status, error
    for entry in merged.values():
        entry["episodes"].sort(key=lambda e: e.get("seed", 0))
    return [merged[job[0]] for job in jobs]


def outcome(episode):
    """Outcome of an episode; supervisor-killed episodes are classified from their last states."""
    if "outcome" in episode:
        return episode["outcome"]
    return "solver_void" if M.void_candidate(episode.get("last_states", [])) else "solver_failure"


def aggregate(case_results):
    """Per-case physical metrics plus `reward`, the success fraction over all scored episodes.

    Void episodes (solver stalls of a plasma within tolerance) are excluded from every
    fraction. Returns (reward dict or None if the void share exceeds VOID_LIMIT, grading).
    An invalid package (policy error in any case, or a median per-step computation above
    LATENCY_LIMIT_MS) scores 0.
    """
    reward, grading_cases = {}, {}
    scored_all, n_all, invalid = [], 0, False
    for r in case_results:
        name = r["case"]
        eps = r["episodes"]
        for e in eps:
            e["outcome"] = outcome(e)
        scored = [e for e in eps if e["outcome"] != "solver_void"]
        feasible = [e for e in scored if e["feasible"]]
        n_success = sum(e["outcome"] == "success" for e in scored)
        entry = {"status": r["status"], "n_episodes": len(eps), "n_void": len(eps) - len(scored),
                 "n_feasible": len(feasible), "n_success": n_success,
                 "outcomes": {o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})}}
        invalid = invalid or r["status"] == "policy_error"
        if r["status"] == "ok" and eps:
            n_all += len(eps)
            scored_all += scored
            reward[f"{name}/void_episodes"] = len(eps) - len(scored)
            if scored:
                reward[f"{name}/success_fraction"] = n_success / len(scored)
                reward[f"{name}/feasible_fraction"] = len(feasible) / len(scored)
            if feasible:
                for key, unit, scale in (("shape_rms", "cm", 100.0), ("xpoint_error", "cm", 100.0),
                                         ("rin_error", "cm", 100.0), ("rout_error", "cm", 100.0),
                                         ("zx_error", "cm", 100.0), ("ip_error_rel", "pct", 100.0)):
                    vals = [e[f"{key}_steady"] for e in feasible if e.get(f"{key}_steady") is not None]
                    if vals:
                        label = "ip_error" if key == "ip_error_rel" else key
                        reward[f"{name}/{label}_{unit}"] = scale * float(np.mean(vals))
            else:
                entry["note"] = "no feasible episode: steady-state metrics not published"
        grading_cases[name] = entry
    computes = [e["policy_compute_ms"] for r in case_results for e in r["episodes"]
                if e.get("policy_compute_ms") is not None]
    if computes:
        reward["policy_compute_ms"] = float(np.median(computes))
        if reward["policy_compute_ms"] > LATENCY_LIMIT_MS:
            invalid = True
            grading_cases["_latency"] = {"note": f"median policy computation {reward['policy_compute_ms']:.3f} ms "
                                                 f"exceeds the {LATENCY_LIMIT_MS:.1f} ms control period"}
    if invalid:
        reward["reward"] = 0.0
    elif n_all and (n_all - len(scored_all)) / n_all > VOID_LIMIT:
        return None, grading_cases
    else:
        reward["reward"] = (sum(e["outcome"] == "success" for e in scored_all) / len(scored_all)
                            if scored_all else 0.0)
    return reward, grading_cases


def default_workers() -> int:
    """One episode per OMP_NUM_THREADS of the CPUs this process may use (cgroup/affinity aware)."""
    if os.environ.get("TOKAGYM_WORKERS"):
        return int(os.environ["TOKAGYM_WORKERS"])
    cpus = len(os.sched_getaffinity(0))
    return max(1, cpus // max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--randomization", default="visible",
                    help="'visible', 'none' or a JSON file with sealed distribution and seeds")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel episodes (default: TOKAGYM_WORKERS, else available CPUs / OMP_NUM_THREADS)")
    ap.add_argument("--policy-timeout", type=float, default=120.0)
    args = ap.parse_args()

    out = Path(args.output)
    if (out / "reward.json").exists():
        raise SystemExit(f"{out}/reward.json already exists; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    randomization, rz_name, default_seeds = load_randomization(args.randomization)
    seeds = args.seeds if args.seeds is not None else default_seeds
    cases = args.cases or sorted(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}")
    scope = "full-suite" if set(cases) == set(CASES) else "development-subset"
    started = time.time()
    jobs = [(c, str(Path(args.submission).resolve()), str(out), randomization, args.policy_timeout)
            for c in cases]
    case_results = run_all(jobs, list(seeds), max(1, args.workers))
    reward, grading_cases = aggregate(case_results)
    infra = [r for r in case_results if r["status"] == "infrastructure_error"]
    results = {
        "env_version": ENV_VERSION, "metric_version": METRIC_VERSION, "protocol": protocol.PROTOCOL,
        "randomization": rz_name, "seeds": seeds, "scope": scope, "cases": case_results,
        "metric_directions": {"reward": "higher", "success_fraction": "higher", "feasible_fraction": "higher",
                              "shape_rms_cm": "lower", "xpoint_error_cm": "lower", "rin_error_cm": "lower",
                              "rout_error_cm": "lower", "zx_error_cm": "lower", "ip_error_pct": "lower"},
        "success_tolerances": {"shape_rms_m": M.SUCCESS_SHAPE_M, "gap_m": M.SUCCESS_GAP_M,
                               "ip_error_rel": M.SUCCESS_IP_REL, "settle_s": M.SETTLE_S},
        "wall_time_sec": time.time() - started,
        "episode_timeout_sec": EPISODE_TIMEOUT_SEC, "step_silence_sec": STEP_SILENCE_SEC,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    grading = {"scope": scope, "randomization": rz_name, "n_cases": len(cases), "n_seeds": len(seeds),
               "aggregate_score": None if reward is None else reward["reward"], "cases": grading_cases,
               "note": "reward = fraction of non-void episodes that keep the plasma within the MAST-U "
                       "steady-state tolerances (METRICS.md); per-case physical metrics alongside."}
    (out / "grading.json").write_text(json.dumps(grading, indent=1))
    if reward is None:
        infra = infra or [{"case": "all", "error": f"more than {VOID_LIMIT:.0%} of episodes were voided by "
                                                    "solver stalls; no score is published"}]
    if infra:
        (out / "error.json").write_text(json.dumps(
            {"infrastructure_errors": {r["case"]: r["error"] for r in infra}}, indent=1))
        print("Infrastructure error; reward.json withheld", file=sys.stderr)
        raise SystemExit(2)
    (out / "reward.json").write_text(json.dumps(harbor_reward(reward), indent=1, allow_nan=False))
    print(json.dumps(reward, indent=1))


if __name__ == "__main__":
    main()
