"""Independent closed-loop replay of a policy package across the frozen case suite.

Usage (inside the declared runtime):
  python evaluation/harbor_verify.py --submission /workspace/submission --output OUT \
      [--cases NAME ...] [--seeds 0 1 2] [--randomization visible|none|FILE] [--workers N]

The evaluator owns the simulator, the perturbation distribution and the metrics; the
policy only supplies coil voltages. Outputs: reward.json (flat numeric, `reward` = success
fraction over every episode), results.json (complete records), grading.json (scope),
per-episode logs.

Robustness (tokamak-shape-metrics/v4): every episode of the suite is scored, so all packages
share one denominator. A control step whose solver raises or exceeds its wall-clock budget is
retried on a fallback ladder (tokagym.env.STEP_NUMERICS) after an exact replay of the episode;
an episode whose worker hangs, crashes or hits a host-side timeout is rerun once with doubled
budgets. Every retry is bounded and recorded. The latency gate is measured after the replay, on
an otherwise idle evaluator, by replaying recorded observations through the package.
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
from tokagym.env import STEP_NUMERICS, VISIBLE_RANDOMIZATION  # noqa: E402,F401
import protocol  # noqa: E402
import supervise  # noqa: E402

METRIC_VERSION = "tokamak-shape-metrics/v4"
DEFAULT_SEEDS = [0, 1, 2]
MAX_RECOVERED_STEPS = 3      # failing steps per episode that get the fallback ladder; later failures end it
MAX_REPLAY_HORIZONS = 4      # replayed steps per episode at most this many horizons: an episode costs at most ~5x its length
BUDGET_SCALES = (1.0, 2.0)   # wall-clock budget multiplier of each episode attempt (one rerun)
TIMING_STEPS = 600           # recorded observations per case replayed for the latency gate


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


# Host-side guards; an episode they stop is rerun, never scored as a control outcome. Up to 4.5 s per
# step for the 1200-step sequence case plus bounded fallback replays; the silence limit exceeds the
# per-step wall-clock hang guard (tokagym.env.STEP_TIME_BUDGET_SEC, 1,800 s); setup sends heartbeats.
EPISODE_TIMEOUT_SEC = float(os.environ.get("TOKAGYM_EPISODE_TIMEOUT", "21600"))
STEP_SILENCE_SEC = float(os.environ.get("TOKAGYM_STEP_SILENCE", "2400"))   # no message for this long -> kill
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


def reset_context(case_name, env, seed):
    return {"environment": case_name, "family": CASES[case_name].family, "seed": seed,
            "machine": env.spec.name, "env_version": ENV_VERSION, "control_dt": 5e-4, "horizon": env.horizon}


def replay_to(env, seed, actions, levels, expected_obs, queue):
    """Reset ``env`` and re-apply the recorded actions (same numerics levels) without the policy.

    The simulator is deterministic (single-threaded, seeded), so the replay must reach exactly the
    observation the policy saw before the failed step; anything else is an infrastructure error.
    """
    obs, _ = env.reset(seed=seed)
    for i, (action, level) in enumerate(zip(actions, levels)):
        env.numerics_level = level
        try:
            obs, _, terminated, _, info = env.step(action)
        finally:
            env.numerics_level = 0
        if terminated:
            raise protocol.InfrastructureError(
                f"replay ended at step {i} ({env.terminated_reason}) where the original episode continued")
        if i % 10 == 0:
            queue.put({"kind": "heartbeat", "replay": True, "shape_rms": info["shape_rms"],
                       "ip_error_rel": info["ip_error_rel"]})
    if not np.array_equal(obs, expected_obs):
        raise protocol.InfrastructureError("replay diverged from the original episode (non-deterministic simulator)")


def recover_step(env, seed, actions, levels, obs_before, queue, failed, budget):
    """Retry the failed last step on the fallback ladder; returns (step result, ladder record).

    ``failed`` is the failed step's result, returned unchanged if no retry fits the episode's replay budget
    (``budget["left"]`` replayed steps, decremented here).
    """
    k = len(actions) - 1
    record = {"step": k, "failure": env.terminated_reason, "failure_kind": env.failure_kind, "levels": []}
    result, recovered = failed, False
    for level in range(1, len(STEP_NUMERICS)):
        if budget["left"] < k:
            record["stopped"] = "replay budget exhausted"
            break
        budget["left"] -= k
        replay_to(env, seed, actions[:k], levels[:k], obs_before, queue)
        env.numerics_level = level
        try:
            result = env.step(actions[k])
        finally:
            env.numerics_level = 0
        levels[k] = level
        recovered = env.failure_kind is None
        record["levels"].append({"level": level, "recovered": recovered,
                                 "failure": None if recovered else env.terminated_reason})
        if recovered:
            break
    record["recovered"] = recovered
    return result, record


def run_case_stream(args, seeds, queue):
    """Child process: replay the given seeds of one case, streaming one message per episode."""
    case_name, submission, out_dir, randomization, timeout, attempt, timing_seed = args
    out = Path(out_dir) / case_name
    out.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(s) for s in seeds)
    policy = null = None
    try:
        env = TokamakMagneticEnv(case_name, randomization=randomization, solver="nonlinear",
                                 strict_actions=True)
        env.budget_scale = BUDGET_SCALES[attempt]
        env.on_progress = lambda: queue.put({"kind": "heartbeat", "setup": True})
        policy = protocol.Policy(submission, out / f"policy-{tag}.stderr", timeout=timeout)
        low = (-env.vmax).tolist()
        high = env.vmax.tolist()
        for seed in seeds:
            started = time.monotonic()
            obs, info = env.reset(seed=seed)
            policy.reset(reset_context(case_name, env, seed), [env.n_obs], [env.n_active], low, high)
            n0 = len(policy.latencies)
            # Diagnostic latency under the replay's load: the null policy answers the same observation
            # next to the submission on every step (the gate itself is measured after the replay). The
            # order alternates: the first process to answer after a solver step pays the wake-up cost.
            null = open_null_policy(case_name, env, out, tag, timeout)
            m0 = len(null.latencies)
            timing = [] if seed == timing_seed else None
            actions, levels, recoveries = [], [], []
            budget = {"left": MAX_REPLAY_HORIZONS * env.horizon}
            ret = 0.0
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                if timing is not None and len(timing) < TIMING_STEPS:
                    timing.append(np.array(obs, dtype=float))
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
                obs_before = obs
                actions.append(np.array(action, dtype=float))
                levels.append(0)
                obs, reward, terminated, truncated, step_info = env.step(action)
                if env.failure_kind == "timeout" and attempt < len(BUDGET_SCALES) - 1:
                    # the wall-clock hang guard is host-side: rerun the episode rather than change its physics
                    raise protocol.InfrastructureError(
                        f"step {len(actions) - 1} of seed {seed} exceeded the wall-clock hang guard")
                if env.failure_kind is not None and len(recoveries) < MAX_RECOVERED_STEPS:
                    (obs, reward, terminated, truncated, step_info), record = recover_step(
                        env, seed, actions, levels, obs_before, queue,
                        (obs, reward, terminated, truncated, step_info), budget)
                    recoveries.append(record)
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
                "kind": "episode", "seed": seed, "return_diagnostic": ret, "attempt": attempt,
                "fallback_steps": sum(1 for level in levels if level > 0),
                "recoveries": recoveries,
                "policy_latency_ms_median": float(np.median(lat) * 1e3) if len(lat) else None,
                "policy_latency_ms_p95": float(np.percentile(lat, 95) * 1e3) if len(lat) else None,
                "null_latency_ms_median": null_ms,
                "policy_compute_ms_under_load": (max(0.0, float(np.median(lat[:n] - null_lat[:n]) * 1e3))
                                                 if n else None),
                "wall_time_sec": time.monotonic() - started,
                "max_step_time_sec": max((r.get("step_time_sec", 0.0) for r in env.records), default=None),
                "slow_steps": sum(r.get("step_time_sec", 0.0) > 60.0 for r in env.records),
                "setup": {k: info.get(k) for k in ("plasma_resistivity", "passive_scale", "supply_delay_steps",
                                                   "reset_attempts", "reset_fallback", "profile_constraint",
                                                   "drift", "crashes", "dead_channels", "handover_steps")},
            })
            if timing:
                np.save(out / "timing-observations.npy", np.stack(timing))
                (out / "timing-context.json").write_text(json.dumps(
                    {"context": reset_context(case_name, env, seed), "n_obs": env.n_obs,
                     "n_active": env.n_active, "low": low, "high": high}))
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


def run_all(jobs, seeds, workers, timing_seed):
    """Run one supervised process per (case, seed); rerun host-side failures once with larger budgets.

    A policy error ends the case (invalid package). An episode stopped by a host-side guard (silence,
    episode time limit), a crashed worker or an evaluator error is rerun with doubled wall-clock
    budgets. If the rerun fails the same way, a stall is scored as a solver failure (it reproduced
    with twice the time) and anything else is an infrastructure error that withholds the score.
    """
    pending = [(job, seed) for job in jobs for seed in seeds]
    merged = {job[0]: {"case": job[0], "family": CASES[job[0]].family, "episodes": [], "status": "ok",
                       "error": None, "supervisor_notes": [], "reruns": 0} for job in jobs}
    rank = {"ok": 0, "policy_error": 1, "infrastructure_error": 2}

    def fail(name, status, error):
        if rank[status] > rank[merged[name]["status"]]:
            merged[name]["status"], merged[name]["error"] = status, error

    for attempt, scale in enumerate(BUDGET_SCALES):
        if not pending:
            break
        last = attempt == len(BUDGET_SCALES) - 1
        raw = supervise.run_supervised(
            run_case_stream, [job + (attempt, timing_seed) for job, _ in pending], [[seed] for _, seed in pending],
            EPISODE_TIMEOUT_SEC * scale, workers, silence_timeout=STEP_SILENCE_SEC * scale)
        retry = []
        for i, (job, seed) in enumerate(pending):
            r, name = raw[i], job[0]
            entry = merged[name]
            done = [e for e in r["episodes"] if "outcome" in e]
            status = r["case"]["status"] if r["case"] is not None else "infrastructure_error"
            error = r["case"]["error"] if r["case"] is not None else "; ".join(r["errors"])[:2000]
            entry["supervisor_notes"] += [f"attempt {attempt}: {e}" for e in r["errors"]]
            if status == "policy_error":
                fail(name, status, error)
            elif done and status == "ok":
                entry["episodes"] += [{k: v for k, v in e.items() if k != "kind"} for e in done]
            elif not last:
                entry["reruns"] += 1
                entry["supervisor_notes"].append(f"attempt {attempt}: seed {seed} rerun ({error[:300]})")
                retry.append((job, seed))
            elif any(e.get("termination") == "solver_timeout" for e in r["episodes"]):
                entry["episodes"].append({"seed": seed, "outcome": "solver_failure", "feasible": False,
                                          "complete": False, "steps": None, "tolerance_time_fraction": 0.0,
                                          "attempt": attempt, "termination": "solver_timeout: the episode "
                                          "stalled again with doubled wall-clock budgets"})
            else:
                fail(name, "infrastructure_error", f"seed {seed}: {error}")
        pending = retry
    for entry in merged.values():
        entry["episodes"].sort(key=lambda e: e.get("seed", 0))
    return [merged[job[0]] for job in jobs]


def measure_latency(cases, submission, out, timeout):
    """The real-time gate, measured after the replay on an otherwise idle evaluator.

    For every case, the observations the package saw in the first TIMING_STEPS steps of one
    episode are replayed through a fresh package process, next to the null policy (alternating
    order, as in the replay). Returns (median per-step computation in ms, per-case medians).
    """
    diffs, per_case = [], {}
    for case in cases:
        path = out / case / "timing-observations.npy"
        if not path.exists():
            continue
        observations = np.load(path)
        meta = json.loads((out / case / "timing-context.json").read_text())
        policy = protocol.Policy(submission, out / case / "policy-timing.stderr", timeout=timeout)
        null = protocol.Policy(NULL_POLICY, out / case / "null-timing.stderr", timeout=timeout)
        try:
            policy.reset(meta["context"], [meta["n_obs"]], [meta["n_active"]], meta["low"], meta["high"])
            null.reset({"environment": case, "family": CASES[case].family, "seed": -1},
                       [meta["n_obs"]], [meta["n_active"]], meta["low"], meta["high"])
            for i, o in enumerate(observations):
                packed = protocol.unpack(o)
                if i % 2 == 0:
                    policy.act(packed)
                try:
                    null.act(packed)
                except protocol.PolicyError as error:
                    raise protocol.InfrastructureError(f"null policy failed: {error}") from error
                if i % 2 == 1:
                    policy.act(packed)
            d = np.array(policy.latencies) - np.array(null.latencies)
            policy.close()
            null.close()
            policy = null = None
        finally:
            if policy is not None:
                policy.abort()
            if null is not None:
                null.abort()
        diffs.append(d)
        per_case[case] = max(0.0, float(np.median(d) * 1e3))
    if not diffs:
        return None, per_case
    return max(0.0, float(np.median(np.concatenate(diffs)) * 1e3)), per_case


def aggregate(case_results, policy_compute_ms=None):
    """Per-case physical metrics plus `reward`, the success fraction over every episode.

    Every episode of the suite is in the denominator (tokamak-shape-metrics/v4); solver failures
    that the fallback ladder could not recover count as failures. Returns (reward dict, grading).
    An invalid package (policy error in any case, or a median per-step computation above
    LATENCY_LIMIT_MS) scores 0.
    """
    reward, grading_cases = {}, {}
    all_eps, invalid = [], False
    for r in case_results:
        name = r["case"]
        eps = r["episodes"]
        feasible = [e for e in eps if e.get("feasible")]
        n_success = sum(e["outcome"] == "success" for e in eps)
        entry = {"status": r["status"], "n_episodes": len(eps), "n_feasible": len(feasible), "n_success": n_success,
                 "outcomes": {o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})},
                 "fallback_steps": sum(e.get("fallback_steps", 0) for e in eps), "reruns": r.get("reruns", 0)}
        invalid = invalid or r["status"] == "policy_error"
        if r["status"] == "ok" and eps:
            all_eps += eps
            reward[f"{name}/success_fraction"] = n_success / len(eps)
            reward[f"{name}/feasible_fraction"] = len(feasible) / len(eps)
            reward[f"{name}/tolerance_time_fraction"] = float(np.mean(
                [e.get("tolerance_time_fraction") or 0.0 for e in eps]))
            reward[f"{name}/solver_failure_episodes"] = sum(e["outcome"] == "solver_failure" for e in eps)
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
    under_load = [e["policy_compute_ms_under_load"] for e in all_eps if e.get("policy_compute_ms_under_load") is not None]
    if under_load:
        reward["policy_compute_ms_under_load"] = float(np.median(under_load))
    if policy_compute_ms is not None:
        reward["policy_compute_ms"] = policy_compute_ms
        if policy_compute_ms > LATENCY_LIMIT_MS:
            invalid = True
            grading_cases["_latency"] = {"note": f"median policy computation {policy_compute_ms:.3f} ms "
                                                 f"exceeds the {LATENCY_LIMIT_MS:.1f} ms control period"}
    if all_eps:
        reward["tolerance_time_fraction"] = float(np.mean([e.get("tolerance_time_fraction") or 0.0 for e in all_eps]))
    reward["solver_failure_episodes"] = sum(e["outcome"] == "solver_failure" for e in all_eps)
    reward["fallback_steps"] = sum(e.get("fallback_steps", 0) for e in all_eps)
    reward["episode_reruns"] = sum(r.get("reruns", 0) for r in case_results)
    reward["episodes"] = len(all_eps)
    if invalid:
        reward["reward"] = 0.0
    else:
        reward["reward"] = sum(e["outcome"] == "success" for e in all_eps) / len(all_eps) if all_eps else 0.0
    return reward, grading_cases


def available_cpus() -> int:
    """CPUs this process may use: the cgroup quota (Docker --cpus) if set, else the affinity mask."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


def default_workers(n_episodes: int) -> int:
    """One single-threaded episode per CPU (two CPUs left for the supervisor and policy processes),
    never more episodes in parallel than the suite has. TOKAGYM_WORKERS overrides."""
    if os.environ.get("TOKAGYM_WORKERS"):
        return int(os.environ["TOKAGYM_WORKERS"])
    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    return max(1, min(n_episodes, (available_cpus() - 2) // threads))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--randomization", default="visible",
                    help="'visible', 'none' or a JSON file with sealed distribution and seeds")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel episodes (default: TOKAGYM_WORKERS, else one per available CPU)")
    ap.add_argument("--policy-timeout", type=float, default=120.0)
    args = ap.parse_args()

    out = Path(args.output)
    if (out / "reward.json").exists():
        raise SystemExit(f"{out}/reward.json already exists; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    randomization, rz_name, default_seeds = load_randomization(args.randomization)
    seeds = list(args.seeds if args.seeds is not None else default_seeds)
    cases = args.cases or sorted(CASES)
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}")
    scope = "full-suite" if set(cases) == set(CASES) else "development-subset"
    workers = args.workers or default_workers(len(cases) * len(seeds))
    submission = str(Path(args.submission).resolve())
    started = time.time()
    jobs = [(c, submission, str(out), randomization, args.policy_timeout) for c in cases]
    case_results = run_all(jobs, seeds, max(1, workers), seeds[0])
    replay_wall = time.time() - started
    policy_compute_ms, latency_per_case = None, {}
    if not any(r["status"] != "ok" for r in case_results):
        try:
            policy_compute_ms, latency_per_case = measure_latency(cases, submission, out, args.policy_timeout)
        except protocol.PolicyError as error:
            for r in case_results:
                r["status"], r["error"] = "policy_error", f"latency measurement: {error}"
        except protocol.InfrastructureError as error:
            for r in case_results:
                r["status"], r["error"] = "infrastructure_error", f"latency measurement: {error}"
    reward, grading_cases = aggregate(case_results, policy_compute_ms)
    infra = [r for r in case_results if r["status"] == "infrastructure_error"]
    results = {
        "env_version": ENV_VERSION, "metric_version": METRIC_VERSION, "protocol": protocol.PROTOCOL,
        "randomization": rz_name, "seeds": seeds, "scope": scope, "cases": case_results,
        "metric_directions": {"reward": "higher", "success_fraction": "higher", "feasible_fraction": "higher",
                              "tolerance_time_fraction": "higher",
                              "shape_rms_cm": "lower", "xpoint_error_cm": "lower", "rin_error_cm": "lower",
                              "rout_error_cm": "lower", "zx_error_cm": "lower", "ip_error_pct": "lower"},
        "success_tolerances": {"shape_rms_m": M.SUCCESS_SHAPE_M, "gap_m": M.SUCCESS_GAP_M,
                               "ip_error_rel": M.SUCCESS_IP_REL, "settle_s": M.SETTLE_S},
        "latency": {"policy_compute_ms": policy_compute_ms, "per_case_ms": latency_per_case,
                    "steps_per_case": TIMING_STEPS, "limit_ms": LATENCY_LIMIT_MS},
        "workers": workers, "replay_wall_time_sec": replay_wall, "wall_time_sec": time.time() - started,
        "episode_timeout_sec": EPISODE_TIMEOUT_SEC, "step_silence_sec": STEP_SILENCE_SEC,
        "budget_scales": BUDGET_SCALES, "max_recovered_steps": MAX_RECOVERED_STEPS,
        "max_replay_horizons": MAX_REPLAY_HORIZONS,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    grading = {"scope": scope, "randomization": rz_name, "n_cases": len(cases), "n_seeds": len(seeds),
               "aggregate_score": None if infra else reward["reward"], "cases": grading_cases,
               "note": "reward = fraction of all episodes that keep the plasma within the MAST-U steady-state "
                       "tolerances (METRICS.md); per-case physical metrics alongside."}
    (out / "grading.json").write_text(json.dumps(grading, indent=1))
    if infra:
        (out / "error.json").write_text(json.dumps(
            {"infrastructure_errors": {r["case"]: r["error"] for r in infra}}, indent=1))
        print("Infrastructure error; reward.json withheld", file=sys.stderr)
        raise SystemExit(2)
    (out / "reward.json").write_text(json.dumps(harbor_reward(reward), indent=1, allow_nan=False))
    print(json.dumps(reward, indent=1))


if __name__ == "__main__":
    main()
