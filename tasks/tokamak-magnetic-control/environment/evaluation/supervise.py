"""Supervised per-case worker processes with a hard per-episode wall-clock limit.

FreeGSNKE can spend unbounded time inside a degenerate Newton-Krylov solve and its
bare ``except:`` clauses can swallow in-process timeouts, so the only guaranteed
guard is external: each case runs in its own process that streams per-episode
results; if an episode exceeds ``episode_timeout`` the process is killed, the
episode is recorded as ``solver_timeout`` (infeasible) and a fresh process resumes
with the remaining seeds.
"""

import multiprocessing as mp
import os
import signal
import time


def _child(target, args, seeds, queue):
    try:
        os.setsid()
    except OSError:
        pass
    try:
        target(args, seeds, queue)
    except BaseException as error:  # noqa: BLE001 - report and exit
        queue.put({"kind": "fatal", "error": f"{type(error).__name__}: {error}"[:2000]})
    finally:
        queue.put({"kind": "exit"})


def _kill(proc):
    if proc.pid is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    proc.join(10)


def run_supervised(target, jobs, seeds, episode_timeout, max_parallel, on_event=None,
                   silence_timeout=None):
    """Run ``target(args, seeds, queue)`` per job with per-episode and per-silence timeouts.

    ``target`` must put {"kind": "episode", "seed": s, ...} after each episode and may
    put {"kind": "heartbeat", ...} between steps and {"kind": "case", ...} once at the end.
    The last heartbeats of a killed episode are attached to it as ``last_states`` so the
    caller can tell a numerical stall of a well-controlled plasma from a control failure.
    A process is killed when it has been silent (no message of any kind) for
    ``silence_timeout`` seconds (default: episode_timeout) or when an episode exceeds
    ``episode_timeout``. ``seeds`` is one list shared by every job or one list per job.
    Returns {job_index: {"episodes": [...], "case": {...} or None, "errors": [...]}}.
    """
    silence_timeout = episode_timeout if silence_timeout is None else silence_timeout
    per_job = bool(seeds) and isinstance(seeds[0], (list, tuple))
    job_seeds = [list(s) for s in seeds] if per_job else [list(seeds)] * len(jobs)
    ctx = mp.get_context("spawn")
    results = {i: {"episodes": [], "case": None, "errors": []} for i in range(len(jobs))}
    pending = list(range(len(jobs)))
    active = {}   # job index -> dict(proc, queue, remaining, started, last)

    def start(i, remaining):
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(target, jobs[i], remaining, q), daemon=True)
        p.start()
        active[i] = {"proc": p, "queue": q, "remaining": list(remaining), "started": time.monotonic(),
                     "last": time.monotonic(), "episode_started": time.monotonic(), "recent": []}

    def last_states(slot):
        return [{k: v for k, v in m.items() if k != "kind"} for m in slot["recent"]]

    while pending or active:
        while pending and len(active) < max_parallel:
            i = pending.pop(0)
            start(i, job_seeds[i])
        for i in list(active):
            slot = active[i]
            drained = False
            while True:
                try:
                    msg = slot["queue"].get_nowait()
                except Exception:  # queue.Empty
                    break
                drained = True
                slot["last"] = time.monotonic()
                if msg["kind"] == "heartbeat":
                    slot["recent"] = (slot["recent"] + [msg])[-20:]
                    continue
                if msg["kind"] == "episode":
                    slot["episode_started"] = time.monotonic()
                    slot["recent"] = []
                    results[i]["episodes"].append(msg)
                    if msg["seed"] in slot["remaining"]:
                        slot["remaining"].remove(msg["seed"])
                    if on_event:
                        on_event(i, msg)
                elif msg["kind"] == "case":
                    results[i]["case"] = msg
                elif msg["kind"] == "fatal":
                    results[i]["errors"].append(msg["error"])
                elif msg["kind"] == "exit":
                    slot["proc"].join(5)
                    del active[i]
                    break
            if i not in active:
                continue
            now = time.monotonic()
            silent = now - slot["last"] > silence_timeout
            too_long = now - slot["episode_started"] > episode_timeout
            # (a steady stream of heartbeats must not defer the per-episode limit)
            if silent or too_long:
                seed = slot["remaining"][0] if slot["remaining"] else None
                _kill(slot["proc"])
                why = (f"silent for {now - slot['last']:.0f}s (> {silence_timeout:.0f}s)" if silent
                       else f"episode exceeded {episode_timeout:.0f}s")
                results[i]["errors"].append(f"episode seed={seed} {why}; process killed")
                if seed is not None:
                    results[i]["episodes"].append({"kind": "episode", "seed": seed, "feasible": False,
                                                   "complete": False, "steps": None,
                                                   "termination": "solver_timeout",
                                                   "last_states": last_states(slot)})
                    slot["remaining"].remove(seed)
                    if on_event:
                        on_event(i, results[i]["episodes"][-1])
                remaining = slot["remaining"]
                del active[i]
                if remaining:
                    start(i, remaining)
                continue
            if not slot["proc"].is_alive() and i in active:
                # died without an exit message (e.g. segfault)
                results[i]["errors"].append("worker process died unexpectedly")
                remaining = slot["remaining"]
                del active[i]
                if remaining:
                    results[i]["episodes"].append({"kind": "episode", "seed": remaining[0], "feasible": False,
                                                   "complete": False, "steps": None,
                                                   "termination": "worker_crash",
                                                   "last_states": last_states(slot)})
                    remaining = remaining[1:]
                    if remaining:
                        start(i, remaining)
        time.sleep(0.5)
    return results
