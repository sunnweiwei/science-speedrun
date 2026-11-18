"""MPI & Multi-Backend Sandbox Verification for fluid-flow-control (#05).

Supports two turnkey GPU verification modes inside the development or verifier container:
  1. `--mode smoke`:
     Runs `env.reset(seed=123)` + 5 zero-action steps across all 26 scored cases
     (`VERIFIER_CASES`: 4 HydroGym [2 JAX + 2 JAX-Fluids] + 22 MAIA), verifying OpenMPI
     container safety (`OMPI_MCA_btl=self,vader,tcp`, `OMPI_MCA_btl_vader_single_copy_mechanism=none`),
     finite observations/rewards, and dual-rank GPU binding (`HYDROGYM_ALLOW_SHARED_GPU`).
  2. `--mode full-dual-rank`:
     Runs full-horizon zero-control rollouts across all 3 dual-rank MAIA cases
     (`Cube_3D_Re3700`, `Cylinder_3D_Re3900`, `NACA0012Gust_3D_Re50000_AOA12`), logging peak GPU
     memory (MiB), wall-clock time (s), and zero-control numerical stability relative to the
     paired zero-control anchors.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = TASK_ROOT / "environment" / "evaluation"
EVAL_DIR = Path(os.environ.get("HYDROGYM_EVALUATOR_CODE", str(DEFAULT_EVAL)))
if not EVAL_DIR.exists():
    EVAL_DIR = DEFAULT_EVAL
sys.path.insert(0, str(EVAL_DIR))

from aggregate import anchors as load_anchors  # noqa: E402
from case_spec import VERIFIER_CASES, local_only, select_cases, verify_resources  # noqa: E402
from evaluate import write_json  # noqa: E402
from maia_gpu_rank import allow_shared_gpu  # noqa: E402

DUAL_RANK_CASES = (
    "Cube_3D_Re3700",
    "Cylinder_3D_Re3900",
    "NACA0012Gust_3D_Re50000_AOA12",
)
SMOKE_STEPS = 5


def cases_for_mode(mode: str, explicit_cases=None):
    if explicit_cases:
        _, selected = select_cases(list(explicit_cases), blind_seeds_path="")
        return selected
    if mode == "smoke":
        _, selected = select_cases(["verifier"], blind_seeds_path="")
        return selected
    if mode == "full-dual-rank":
        _, selected = select_cases(list(DUAL_RANK_CASES), blind_seeds_path="")
        return selected
    raise ValueError(f"Unsupported mode: {mode}")


def query_gpu_memory_mib():
    """Query per-GPU used/total memory in MiB via nvidia-smi without initializing CUDA in parent."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                rows.append({
                    "uuid": parts[0],
                    "used_mib": float(parts[1]),
                    "total_mib": float(parts[2]),
                })
            except ValueError:
                continue
    return rows


class PeakGpuMemoryMonitor:
    """Background poller tracking peak GPU memory usage during a CFD subprocess."""

    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.peak_by_uuid = {}
        self.total_by_uuid = {}

    def _poll_once(self):
        for row in query_gpu_memory_mib():
            uid = row["uuid"]
            self.peak_by_uuid[uid] = max(self.peak_by_uuid.get(uid, 0.0), row["used_mib"])
            self.total_by_uuid[uid] = row["total_mib"]

    def start(self):
        self._poll_once()

        def loop():
            while not self._stop.wait(self.interval_sec):
                self._poll_once()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._poll_once()
        return {
            "peak_used_mib_by_gpu": dict(self.peak_by_uuid),
            "total_mib_by_gpu": dict(self.total_by_uuid),
            "max_peak_used_mib": max(self.peak_by_uuid.values()) if self.peak_by_uuid else None,
        }


def run_worker_rollout(case, data: Path, output: Path, steps: int, full_horizon: bool):
    import numpy as np
    from native_envs import ReplayAdapter, make_env

    env = make_env(case, data)
    adapter = ReplayAdapter(env, case, output)
    can_close = True
    started = time.monotonic()
    try:
        obs, _ = adapter.reset(seed=123)
        obs_arr = np.asarray(obs, dtype=np.float64)
        if not np.all(np.isfinite(obs_arr)):
            raise RuntimeError("Nonfinite initial observation on reset(seed=123)")
        zero_action = np.clip(
            np.zeros(adapter.action_space.shape, dtype=adapter.action_space.dtype),
            adapter.action_space.low,
            adapter.action_space.high,
        )
        rewards = []
        terminated = truncated = False
        target_steps = case["horizon"] if full_horizon else min(steps, case["horizon"])
        for step_idx in range(target_steps):
            obs, reward, terminated, truncated, _ = adapter.step(zero_action)
            if not np.all(np.isfinite(np.asarray(obs))) or not math.isfinite(float(reward)):
                raise RuntimeError(f"Nonfinite state/reward at step {step_idx + 1}")
            rewards.append(float(reward))
            if terminated or truncated:
                break
        record = {
            "status": "ok",
            "case": case["id"],
            "backend": case["backend"],
            "solver_processes": case["solver_processes"],
            "steps_executed": len(rewards),
            "target_steps": target_steps,
            "return_sum": float(math.fsum(rewards)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "worker_wall_sec": time.monotonic() - started,
        }
        if full_horizon:
            if len(rewards) != case["horizon"] or not (terminated or truncated):
                raise RuntimeError(
                    f"Full-horizon rollout ended at {len(rewards)}/{case['horizon']} steps"
                )
            pm = adapter.evaluation_metrics()
            if pm.get("status") != "ok" or not math.isfinite(float(pm["value"])):
                raise RuntimeError(f"Invalid primary metric on zero-control rollout: {pm}")
            ref_zero = float(load_anchors()[case["id"]]["zero"])
            val = float(pm["value"])
            rel_diff = abs(val - ref_zero) / abs(ref_zero) if ref_zero != 0 else abs(val)
            record["primary_metric"] = pm
            record["zero_control_stability"] = {
                "measured_zero": val,
                "reference_zero_anchor": ref_zero,
                "abs_diff": abs(val - ref_zero),
                "rel_diff": rel_diff,
                "finite_stable": True,
            }
        write_json(output / "worker-result.json", record)
    except BaseException:
        can_close = case["backend"] != "maia"
        raise
    finally:
        if can_close:
            env.close()


def launch_single_case(case, data: Path, case_out: Path, mode: str, steps: int):
    case_out.mkdir(parents=True, exist_ok=True)
    helper = EVAL_DIR / "maia_gpu_rank.py"
    devices = json.loads(subprocess.check_output([sys.executable, str(helper), "--list-gpus"], text=True))
    required = case["solver_processes"]
    shared_gpu = False
    if len(devices) < required:
        if case["backend"] == "maia" and len(devices) >= 1 and allow_shared_gpu():
            selected = list(devices)
            shared_gpu = True
        else:
            raise RuntimeError(
                f"{case['id']} requires {required} GPUs; visible={len(devices)} "
                f"(HYDROGYM_ALLOW_SHARED_GPU={os.environ.get('HYDROGYM_ALLOW_SHARED_GPU', '1')})"
            )
    else:
        selected = devices[:required]

    child_env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(selected),
        HYDROGYM_GPU_BINDING_DIR=str(case_out),
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_coll_hcoll_enable=os.environ.get("OMPI_MCA_coll_hcoll_enable", "0"),
        OMPI_MCA_btl=os.environ.get("OMPI_MCA_btl", "self,vader,tcp"),
        OMPI_MCA_btl_vader_single_copy_mechanism=os.environ.get(
            "OMPI_MCA_btl_vader_single_copy_mechanism", "none"
        ),
    )

    runtime_sh = EVAL_DIR / "native_runtime.sh"
    script = Path(__file__).resolve()
    if case["backend"] == "maia":
        cmd = [
            "bash",
            str(runtime_sh),
            "maia",
            "python",
            str(script),
            "--maia-launcher",
            "--mode",
            mode,
            "--steps",
            str(steps),
            "--case",
            case["id"],
            "--data",
            str(data),
            "--output",
            str(case_out),
        ]
    else:
        cmd = [
            "bash",
            str(runtime_sh),
            case["backend"],
            "python",
            str(script),
            "--worker",
            "--mode",
            mode,
            "--steps",
            str(steps),
            "--case",
            case["id"],
            "--data",
            str(data),
            "--output",
            str(case_out),
        ]

    monitor = PeakGpuMemoryMonitor()
    monitor.start()
    started = time.monotonic()
    with (case_out / "rollout.log").open("wb") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
    wall_sec = time.monotonic() - started
    gpu_mem = monitor.stop()

    if proc.returncode != 0 or not (case_out / "worker-result.json").is_file():
        return {
            "status": "failed",
            "case": case["id"],
            "backend": case["backend"],
            "solver_processes": required,
            "shared_gpu": shared_gpu,
            "allocated_gpus": selected,
            "exit_code": proc.returncode,
            "wall_sec": wall_sec,
            "gpu_memory": gpu_mem,
            "log": str(case_out / "rollout.log"),
        }

    res = json.loads((case_out / "worker-result.json").read_text())
    res.update(
        shared_gpu=shared_gpu,
        allocated_gpus=selected,
        wall_sec=wall_sec,
        gpu_memory=gpu_mem,
    )
    return res


def run_maia_mpirun(case, data: Path, case_out: Path, mode: str, steps: int):
    local_only(data, case, case_out)
    from hydrogym.maia.workspace import prepare_maia_workspace

    work, props = prepare_maia_workspace(
        case["environment"],
        work_dir=str(case_out / "cfd"),
        use_clean_cache=False,
        local_fallback_dir=str(data / "environments"),
    )
    solver = [sys.executable, str(EVAL_DIR / "maia_gpu_rank.py")]
    cmd = [
        "mpirun",
        "--oversubscribe",
        "--bind-to",
        "none",
        "-np",
        "1",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--steps",
        str(steps),
        "--case",
        case["id"],
        "--data",
        str(data),
        "--output",
        str(case_out),
        ":",
        "-np",
        str(case["solver_processes"]),
        *solver,
        props,
        "--silent",
    ]
    subprocess.run(cmd, cwd=work, env=dict(os.environ), check=True)


def run_suite(mode: str, data: Path, output: Path, steps: int = SMOKE_STEPS,
              explicit_cases=None, runner=launch_single_case, verify_data: bool = True):
    cases = cases_for_mode(mode, explicit_cases)
    if verify_data:
        verify_resources(data.resolve(), cases)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    started = time.monotonic()
    for case in cases:
        rows[case["id"]] = runner(case, data.resolve(), output / case["id"], mode, steps)
    passed = sum(1 for r in rows.values() if r.get("status") == "ok")
    hydrogym_count = sum(1 for c in cases if c["backend"] != "maia")
    maia_count = sum(1 for c in cases if c["backend"] == "maia")
    summary = {
        "schema": "hydrogym-mpi-check/v1",
        "mode": mode,
        "status": "ok" if passed == len(cases) else "failed",
        "passed_cases": passed,
        "total_cases": len(cases),
        "hydrogym_jax_cases": hydrogym_count,
        "maia_cases": maia_count,
        "allow_shared_gpu": allow_shared_gpu(),
        "total_wall_sec": time.monotonic() - started,
        "cases": rows,
    }
    filename = "smoke_summary.json" if mode == "smoke" else "full_dual_rank_summary.json"
    write_json(output / filename, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "full-dual-rank"], default="smoke")
    p.add_argument("--data", type=Path, default=Path("/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=SMOKE_STEPS,
                   help="Number of steps in smoke mode (default: 5)")
    p.add_argument("--case", action="append", help="Optional explicit case filter")
    p.add_argument("--maia-launcher", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.maia_launcher:
        _, selected = select_cases([args.case[0]], blind_seeds_path="")
        run_maia_mpirun(selected[0], args.data.resolve(), args.output.resolve(), args.mode, args.steps)
        return

    if args.worker:
        _, selected = select_cases([args.case[0]], blind_seeds_path="")
        case = selected[0]
        local_only(args.data.resolve(), case, args.output.resolve())
        try:
            run_worker_rollout(
                case,
                args.data.resolve(),
                args.output.resolve(),
                steps=args.steps,
                full_horizon=(args.mode == "full-dual-rank"),
            )
        except BaseException:
            if case["backend"] == "maia":
                from mpi4py import MPI
                MPI.COMM_WORLD.Abort(1)
            raise
        return

    summary = run_suite(args.mode, args.data, args.output, steps=args.steps, explicit_cases=args.case)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
