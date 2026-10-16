"""Manual MAIA baseline reconstruction, never a certified paper score."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from time_limits import apply_time_limit


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime_properties(source, output, max_cells):
    """Optional private allocation-capacity override; never edit source assets."""
    source = Path(source).resolve()
    if max_cells is None:
        return source, None
    if type(max_cells) is not int or max_cells <= 0:
        raise ValueError("max-cells must be a positive integer allocation capacity")
    payload = source.read_bytes()
    original = payload.decode()
    pattern = r"(?m)^(maxNoCells\s*=[ \t]*)([1-9][0-9]*)([ \t]*(?:#.*)?)$"
    matches = list(re.finditer(pattern, original))
    if len(matches) != 1:
        raise ValueError("Expected one unambiguous maxNoCells property")
    changed = re.sub(pattern, lambda match: match[1] + str(max_cells) + match[3], original)
    destination = Path(output) / "runtime-properties.toml"
    # Refuse overwriting either a prior runtime configuration or a source link.
    with destination.open("x") as stream:
        stream.write(changed)
    if source.read_bytes() != payload:
        raise RuntimeError("Published properties changed while copying")
    return destination.resolve(), {"kind": "allocation-capacity-only-not-mesh-coarsening",
        "source_path": str(source), "source_sha256": hashlib.sha256(payload).hexdigest(),
        "runtime_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "maxNoCells": {"original": int(matches[0][2]), "effective": max_cells}}


def author_pinball_3d_probes():
    """Public torchRL_examples/ppo.py 3D meshgrid, not final-figure metadata."""
    import numpy as np
    # np.meshgrid(x, y, z)'s default xy indexing flattens in y,x,z order.
    return [(float(x), float(y), float(z))
            for y in np.linspace(-.75, .75, 5)
            for x in np.linspace(2., 8., 7)
            for z in np.linspace(-1.5, 1.5, 5)]


def author_cylinder_1000_probes():
    """Author CylinderBase (1000,2) defaults; not recovered figure metadata."""
    import numpy as np
    # Preserve the public meshgrid's y-major ordering. No cached normalization.
    return [(float(x), float(y)) for y in np.linspace(-.75, .75, 5)
            for x in np.linspace(1., 7., 13)]


def author_square_200_probes():
    """Author SquareCylinderBase (200,2) sensor grid, in meshgrid order."""
    import numpy as np
    return [(float(x), float(y)) for y in np.linspace(-.75, .75, 5)
            for x in np.linspace(1., 8., 8)]


def paper_coupling(case):
    """SI case-specific (lift penalty, CFD substeps, control horizon)."""
    return {
        "Pinball_2D_Re100": (1., 170, 200),
        "Pinball_3D_Re150": (1., 200, 200),
        "Cylinder_2D_Re3900": (1., 600, 200),
        "Cylinder_2D_Re1000": (1., 350, 200),
        "SquareCylinder_2D_Re200": (.2, 225, 200),
        "NACA0012Gust_3D_Re1000_AOA20": (.25, 640, 25),
    }[case]


def logged_force_coefficients(env, case, forces):
    """Normalize diagnostics exactly as the native reward, without changing it."""
    import numpy as np
    force = np.asarray(forces)
    if case.startswith(("Cylinder_", "SquareCylinder_", "NACA")):
        force = env.compute_nondim_coefficients(force, density=1.0,
            referenceVelocity=env.Ma / np.sqrt(3),
            projectionLength=env.referenceLength / env.dX * env.zLength / env.dX)
    return np.atleast_2d(force)


def release_identity(data, case):
    """Bind new runs to a selected release; keep legacy archives replayable."""
    path = Path(data) / "release.json"
    if not path.exists():
        return {}
    release = json.loads(path.read_text())
    receipt = json.loads((Path(data) / "materialized.json").read_text())
    digest = hashlib.sha256(json.dumps(release, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (receipt.get("data_release_id") != release["release_id"] or
            receipt.get("data_release_sha256") != digest):
        raise ValueError("Source release/receipt mismatch")
    matches = [item for item in release["cases"] if item["id"] == case]
    if len(matches) != 1 or case not in release["training"]["case_ids"]:
        raise ValueError("Case is outside the selected data release")
    return {"data_release_id": release["release_id"], "data_release_sha256": digest,
            "case_files_sha256": matches[0]["files_sha256"]}


def local_resources(args):
    import hydrogym.data_manager as manager
    # Only alter resource discovery; do not change the solver implementation.
    manager.HF_AVAILABLE = False
    source = args.data / "environments" / args.environment
    receipt = json.loads((args.data / "materialized.json").read_text())
    if receipt["revision"] != "cf5ef6748443ae72e9647f398487e98d421a7243" or args.environment not in receipt["verified_directories"]:
        raise RuntimeError("Missing verified source resources")
    release_identity(args.data, args.environment)
    cached = Path.home() / ".cache/maiagym" / args.environment
    if cached.exists() and cached.resolve() != source.resolve():
        raise RuntimeError("Legacy cache shadows pinned assets")
    return source


def environment(args):
    import gymnasium as gym
    import hydrogym.maia as maia
    import numpy as np

    source = local_resources(args)
    # Explicitly inferred wake probes, not claimed to be the paper's exact list.
    probes = [(float(x), float(y)) for x in np.linspace(0.75, 4.25, 8)
              for y in np.linspace(-1.75, 1.75, 8)]
    if args.protocol == "author-paper":
        if args.environment == "Pinball_2D_Re100":
            # Case-specific defaults in the author's Pinball2DRe100 class.
            probes = [(float(x), float(y)) for x in np.linspace(1.5, 7.5, 7)
                      for y in np.linspace(-1.5, 1.5, 7)]
        elif args.environment == "Pinball_3D_Re150":
            probes = author_pinball_3d_probes()
        elif args.environment == "Cylinder_2D_Re3900":
            # Author CylinderBase default_values[(3900, 2)], not claimed to be
            # the final figure sensors. Actual public mesh is cylinder-centred.
            probes = [(float(x), float(y)) for x in np.linspace(1.0, 8.0, 4)
                      for y in np.linspace(-0.75, 0.75, 3)]
        elif args.environment == "Cylinder_2D_Re1000":
            probes = author_cylinder_1000_probes()
        elif args.environment == "SquareCylinder_2D_Re200":
            probes = author_square_200_probes()
        elif args.environment == "NACA0012Gust_3D_Re1000_AOA20":
            # Public author default_values[(1000,3)], not certified figure sensors.
            probes = [(float(x), float(y)) for y in np.linspace(-.75, .75, 3)
                      for x in np.linspace(1., 8., 4)]
        else:
            raise ValueError("author-paper sensor reconstruction not audited for this case")
    if "_3D_" in args.environment and len(probes[0]) == 2:
        # The supplied span is [-2, 2]; keep probes inside the domain.
        levels = (-.75, 0., .75) if args.environment.startswith("NACA") else (-1., 0., 1.)
        probes = [(x, y, z) for z in levels for x, y in probes]
    normalization = {"obs_normalization_strategy": "U_inf"}
    normalization_audit = None
    if args.normalization == "author":
        from pinball_author import load
        probes, vectors, normalization_audit = load(args.author_source)
        normalization = {"obs_normalization_strategy": "customized",
                         **{k: v.tolist() for k, v in vectors.items()}}
    probes = [coordinate for point in probes for coordinate in point]
    env = maia.from_hf(args.environment, use_clean_cache=False,
                       local_fallback_dir=str(source.parent), probe_locations=probes,
                       **normalization)
    if Path(env.env_data_path).resolve() != source.resolve():
        raise RuntimeError("Unexpected resource source")
    raw = {"omega": float(env.omega), "substeps": int(env.num_substeps_per_iteration),
           "max_episode_steps": int(env.max_episode_steps), "max_control": float(env.MAX_CONTROL),
           "max_refinement_level": int(env.maxRfnmntLvl), "dx": float(env.dX)}
    horizon = env.max_episode_steps + 1
    if args.protocol in ("paper-reconstruction", "author-paper"):
        env.omega, env.num_substeps_per_iteration, horizon = paper_coupling(args.environment)
        # Native completion uses >; this reproduces the stated 200 actions.
        env.max_episode_steps = horizon - 1
    if args.protocol == "author-paper":
        if args.max_control is not None:
            env.MAX_CONTROL = args.max_control
        elif args.environment == "Pinball_2D_Re100":
            env.MAX_CONTROL = 0.5
    if args.diagnostic_horizon is not None:
        horizon = args.diagnostic_horizon
        env.max_episode_steps = horizon - 1
    if args.bootstrap_time_limits:
        # Public native completion is iter > limit; StepCounter truncates first.
        env.max_episode_steps = horizon
    provenance = {"status": "pilot-not-certified-paper-reproduction", "environment": args.environment,
                  "protocol": args.protocol, "native_defaults": raw, "effective_omega": float(env.omega),
                  "effective_substeps": int(env.num_substeps_per_iteration), "horizon": horizon,
                  "time_limit_bootstrap": args.bootstrap_time_limits,
                  "effective_max_control": float(env.MAX_CONTROL),
                  "diagnostic_horizon_override": args.diagnostic_horizon,
                  "probe_locations": probes, "observation_normalization": "U_inf",
                  "observation_size": int(np.prod(env.observation_space.shape)),
                  "observation_type": list(env.observation_type),
                  "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
                  "unresolved": ["exact paper probes", "mesh equivalence", "original learner configuration"],
                  "seed": args.seed, "source_path": str(source)}
    provenance.update(release_identity(args.data, args.environment))
    if args.environment == "SquareCylinder_2D_Re200":
        provenance["actuation_source"] = "pinned Hub independent jets; SI 5D, not old paired-jet author class"
        provenance["unresolved"].extend([
            "final-figure jet locations and amplitude; Hub has two inputs at .03",
            "old author class has one paired input at .066 and 87 substeps; not used",
        ])
    if normalization_audit:
        provenance.update(observation_normalization="author-cached", normalization_source=normalization_audit,
                          observation_loc=normalization["obs_loc"], observation_scale=normalization["obs_scale"])
    if getattr(args, "max_cells", None) is not None:
        provenance["solver_max_cells"] = args.max_cells
    write(args.output / "protocol.json", provenance)
    original_reward = env.get_reward
    last_metrics = {}

    def recorded_reward():
        reward, info = original_reward()
        force = logged_force_coefficients(env, args.environment, info["forces"])
        last_metrics.update(drag=float(force[:, 0].sum()), lift=float(force[:, 1].sum()),
                            force_coefficients=force.tolist())
        return reward, info

    env.get_reward = recorded_reward

    class Logged(gym.Wrapper):
        def __init__(self):
            super().__init__(env)
            self.episode = 0
            self.phase = "training" if args.mode == "train" else "evaluation"
            self.training_step = 0
            self.log = (args.output / "episodes.jsonl").open("a", buffering=1)
            self.trajectory = (args.output / "evaluation-steps.jsonl").open("a", buffering=1)

        def reset(self, **kwargs):
            self.episode += 1
            self.episode_started = time.monotonic()
            self.count, self.total = 0, 0.0
            self.drags, self.lifts, self.controls = [], [], []
            obs, info = self.env.reset(**kwargs)
            with np.errstate(over="ignore", invalid="ignore"):
                obs = np.asarray(obs, dtype=np.float32)
            if not np.isfinite(obs).all():
                raise RuntimeError("Nonfinite CFD reset")
            self.previous_observation = obs.astype(np.float32).copy()
            return obs.astype(np.float32), info

        def step(self, action):
            action = np.asarray(action, dtype=self.action_space.dtype)
            if not np.isfinite(action).all() or not self.action_space.contains(action):
                raise RuntimeError("Invalid control action before CFD step")
            step_started = time.monotonic()
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_elapsed = time.monotonic() - step_started
            # Validate the precision actually handed to the policy as well.
            # Finite float64 solver values may overflow a float32 observation.
            with np.errstate(over="ignore", invalid="ignore"):
                obs = np.asarray(obs, dtype=np.float32)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                write(args.output / "transition-error.json", {
                    "episode": self.episode, "phase": self.phase,
                    "control_in_episode": self.count + 1, "training_step": self.training_step,
                    "normalized_action": action.tolist(),
                    "physical_control": (action * env.MAX_CONTROL).tolist(),
                    "observation_nonfinite_count": int((~np.isfinite(obs)).sum()),
                    "reward_repr": repr(float(reward))})
                raise RuntimeError("Nonfinite CFD trajectory")
            self.count += 1
            terminated, truncated = apply_time_limit(terminated, truncated, self.count,
                                                      horizon, args.bootstrap_time_limits)
            self.total += float(reward)
            self.drags.append(last_metrics["drag"])
            self.lifts.append(last_metrics["lift"])
            self.controls.append(float(np.square(action * env.MAX_CONTROL).sum()))
            if self.phase == "evaluation":
                self.trajectory.write(json.dumps({
                    "episode": self.episode, "step": self.count, "training_step": self.training_step,
                    "reward": float(reward), "drag": self.drags[-1], "lift": self.lifts[-1],
                    "normalized_action": np.asarray(action).tolist(),
                    "physical_control": (np.asarray(action) * env.MAX_CONTROL).tolist(),
                    "observation": self.previous_observation.tolist(),
                    "next_observation": obs.astype(np.float32).tolist(),
                    "force_coefficients": last_metrics["force_coefficients"],
                    **({"cfd_step_wall_seconds": step_elapsed}
                       if getattr(args, "record_step_times", False) else {}),
                }, allow_nan=False) + "\n")
            self.previous_observation = obs.astype(np.float32).copy()
            if terminated or truncated:
                record = {"episode": self.episode, "steps": self.count, "return": self.total,
                          "terminated": terminated, "truncated": truncated,
                          "phase": self.phase, "training_step": self.training_step,
                          "mean_drag": float(np.mean(self.drags)), "mean_lift": float(np.mean(self.lifts)),
                          "rms_lift": float(np.sqrt(np.mean(np.square(self.lifts)))),
                          "mean_squared_control_proxy": float(np.mean(self.controls)),
                          "last_half_mean_drag": float(np.mean(self.drags[self.count // 2:])),
                          "last_half_rms_lift": float(np.sqrt(np.mean(np.square(self.lifts[self.count // 2:])))),
                          "wall_sec": time.monotonic() - self.episode_started}
                self.log.write(json.dumps(record, allow_nan=False) + "\n")
                print("EPISODE", json.dumps(record, allow_nan=False), flush=True)
            return obs.astype(np.float32), float(reward), terminated, truncated, info

        def close(self):
            self.log.close()
            self.trajectory.close()
            self.env.close()

    return Logged(), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "zero", "fixed", "train", "evaluate", "replay"])
    parser.add_argument("--environment", choices=["Pinball_2D_Re100", "Pinball_3D_Re150", "Cylinder_2D_Re3900", "Cylinder_2D_Re1000", "SquareCylinder_2D_Re200",
                                                  "NACA0012Gust_3D_Re1000_AOA20"], default="Pinball_2D_Re100")
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["asset-native", "paper-reconstruction", "author-paper"], default="asset-native")
    parser.add_argument("--learner", choices=["pilot", "public-source"], default="pilot")
    parser.add_argument("--recipe", default="public")
    parser.add_argument("--normalization", choices=["u-inf", "author"], default="u-inf")
    parser.add_argument("--author-source", type=Path, default=Path(os.environ.get("HYDROGYM_AUTHOR_SOURCE", "/author-source")))
    parser.add_argument("--eval-every", type=int, default=4000)
    parser.add_argument("--bootstrap-time-limits", action="store_true",
                        help="Public StepCounter semantics: truncate at the horizon and bootstrap value")
    parser.add_argument("--max-control", type=float, choices=[0.03, 0.25, 0.5])
    parser.add_argument("--diagnostic-horizon", type=int,
                        help="Explicit non-paper continuous zero/fixed-control window")
    parser.add_argument("--fixed-action", type=float, nargs="+", default=[0.0, -1.0, 1.0])
    parser.add_argument("--evaluate-published", action="store_true",
                        help="After public-source training, evaluate final/best policies in fresh MPI processes")
    parser.add_argument("--algo", choices=["PPO", "DDPG", "TD3"], default="PPO")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--evaluator-code", type=Path,
                        default=Path(os.environ.get("HYDROGYM_EVALUATOR_CODE", str(Path(__file__).with_name("evaluator")))))
    parser.add_argument("--solver-processes", type=int, default=1)
    parser.add_argument("--max-cells", type=int,
                        help="Explicit per-rank allocation capacity in a private runtime properties copy; never changes the supplied grid")
    parser.add_argument("--record-step-times", action="store_true",
                        help="Diagnostic timing of each evaluation CFD step, excluding policy inference and reset")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.environment.startswith("NACA") and (args.mode not in ("smoke", "zero") or args.max_control is not None):
        parser.error("NACA currently supports only zero/smoke diagnostics at the source actuator limit")
    if args.normalization == "author" and (args.environment != "Pinball_2D_Re100" or args.protocol != "author-paper"):
        parser.error("Author cached normalization is audited only for Pinball_2D_Re100/author-paper")
    if args.max_control is not None and args.environment.startswith(("Cylinder_", "SquareCylinder_")) and args.max_control != 0.03:
        parser.error("Cylinder reconstruction retains the published 0.03 actuator limit")
    if args.max_control == 0.03 and not args.environment.startswith(("Cylinder_", "SquareCylinder_")):
        parser.error("0.03 actuator limit is only audited for the cylinder case")
    if args.diagnostic_horizon is not None and (args.mode not in ("zero", "fixed") or args.diagnostic_horizon <= 0):
        parser.error("Diagnostic horizon is positive and only valid for zero/fixed controls")
    if args.evaluate_published and (args.mode != "train" or args.learner != "public-source"):
        parser.error("--evaluate-published requires public-source train")
    if min(args.steps, args.episodes, args.solver_processes) <= 0:
        parser.error("steps, episodes and solver-processes must be positive")
    if args.max_cells is not None and args.max_cells <= 0:
        parser.error("max-cells must be positive")
    if args.mode == "smoke" and args.steps > 20:
        parser.error("smoke requires an explicit --steps value between 1 and 20")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import numpy as np
        env, provenance = environment(args)
        try:
            if args.mode == "smoke":
                started = time.monotonic()
                env.reset(seed=args.seed)
                for _ in range(args.steps):
                    _, _, done, truncated, _ = env.step(np.zeros(env.action_space.shape))
                    if done or truncated:
                        raise RuntimeError("Unexpected early termination during CFD diagnostic")
                write(args.output / "diagnostic.json", {
                    "status": "finite-native-CFD-steps-only-not-a-benchmark-score",
                    "steps": args.steps, "wall_sec": time.monotonic() - started})
            elif args.mode in ("zero", "fixed"):
                for i in range(args.episodes):
                    env.reset(seed=args.seed + i)
                    for _ in range(provenance["horizon"]):
                        action = np.zeros(env.action_space.shape) if args.mode == "zero" else np.asarray(args.fixed_action, dtype=np.float32)
                        _, _, done, truncated, _ = env.step(action)
                        if done or truncated:
                            break
                write(args.output / "completed.json", {"status": f"native-{args.mode}-control-validation", "episodes": args.episodes})
            elif args.mode == "replay":
                import shutil
                sys.path.insert(0, str(args.evaluator_code))
                from evaluate import package_hash
                from protocol import Policy, replay_episode
                if args.package is None:
                    raise ValueError("replay requires a --package")
                snapshot = args.output / "relocated-policy"
                original_hash = package_hash(args.package)
                shutil.copytree(args.package, snapshot)
                if package_hash(snapshot) != original_hash:
                    raise RuntimeError("Policy changed during snapshot")
                policy = Policy(snapshot, args.output / "policy.stderr", timeout=120)
                try:
                    case = {"environment": args.environment, "family": args.environment.split("_")[0].lower(),
                            "horizon": provenance["horizon"]}
                    rows = [replay_episode(env, policy, case, args.seed + i) for i in range(args.episodes)]
                    policy.close()
                finally:
                    policy.abort()
                if package_hash(snapshot) != original_hash:
                    raise RuntimeError("Policy package mutated during replay")
                write(args.output / "completed.json", {"status": "relocated-reference-replay-not-full-suite",
                      "episodes": rows, "package_hash": original_hash})
            else:
                if args.learner == "public-source":
                    if args.protocol != "author-paper":
                        raise ValueError("public-source requires --protocol author-paper")
                    from public_learner import run
                else:
                    from learner import run
                run(env, args, provenance)
        except BaseException:
            # A failure inside the native reward can interrupt the MPI data
            # exchange. Normal finishRun then waits on the wrong message tag.
            # Preserve the exception and abort this experiment's ranks instead.
            detail = traceback.format_exc()
            write(args.output / "worker-error.json", {"traceback": detail,
                "episode": env.episode, "training_step": env.training_step})
            print(detail, file=sys.stderr, flush=True)
            env.log.flush()
            env.trajectory.flush()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
            raise
        else:
            env.close()
        return
    local_resources(args)
    from hydrogym.maia.workspace import prepare_maia_workspace
    work, props = prepare_maia_workspace(args.environment, work_dir=str(args.output / "cfd"),
                                         local_fallback_dir=str(args.data / "environments"), use_clean_cache=False)
    props, capacity_audit = runtime_properties(props, args.output, args.max_cells)
    command = ["mpirun", "--bind-to", "none", "-np", "1", sys.executable, str(Path(__file__).resolve()),
               *sys.argv[1:], "--worker", ":", "-np", str(args.solver_processes), "maia", props, "--silent"]
    command = [str(item) for item in command]
    launch = {"command": command, "cwd": work}
    if capacity_audit is not None:
        launch["allocation_capacity_override"] = capacity_audit
    write(args.output / "launch.json", launch)
    environment_variables = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    subprocess.run(command, cwd=work, env=environment_variables, check=True)
    if args.evaluate_published:
        # Training MPI workers have exited. New workers reload the immutable
        # source assets and checkpoint, with no training state or replay buffer.
        import hashlib
        seen = set()
        for label, name in (("final", "candidate.pt"), ("best", "best.pt")):
            checkpoint = args.output / name
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            evaluation_command = [sys.executable, str(Path(__file__).resolve()), "evaluate",
                            "--learner", "public-source", "--protocol", args.protocol,
                            "--environment", args.environment, "--algo", args.algo,
                            "--seed", str(args.seed), "--episodes", "1", "--data", str(args.data),
                            "--author-source", str(args.author_source), "--checkpoint", str(checkpoint),
                            "--normalization", args.normalization,
                            "--solver-processes", str(args.solver_processes),
                            "--output", str(args.output / f"independent-{label}")]
            if args.max_control is not None:
                evaluation_command.extend(["--max-control", str(args.max_control)])
            if args.bootstrap_time_limits:
                evaluation_command.append("--bootstrap-time-limits")
            if args.max_cells is not None:
                evaluation_command.extend(["--max-cells", str(args.max_cells)])
            if args.record_step_times:
                evaluation_command.append("--record-step-times")
            subprocess.run(evaluation_command, env=environment_variables, check=True)
        write(args.output / "workflow-completed.json", {
            "status": "trained-and-independently-evaluated-not-paper-certified", "policy_sha256": sorted(seen)})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if "--worker" in sys.argv:
            from mpi4py import MPI
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
