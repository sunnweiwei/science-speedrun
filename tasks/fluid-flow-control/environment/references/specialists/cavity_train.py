"""Native cavity reference training with public author probes and Eq.36.

One environment-specific reference, not the task's shared-model submission.
Uses a measured Re4200 restart, never silently relabels the public Re4140 state.
The figure-specific calibration and learning configuration remain unrecovered.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from time_limits import apply_time_limit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "zero", "replay"])
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--author-source", type=Path, default=Path(os.environ.get("HYDROGYM_AUTHOR_SOURCE", "/author-source")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--restart-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Optional reconstructed target; otherwise published author vectors")
    parser.add_argument("--algo", choices=["PPO", "DDPG", "TD3"], default="PPO")
    parser.add_argument("--recipe", default="public")
    parser.add_argument("--max-control", type=float, choices=[.02, .075], default=.075,
                        help=".075 public author example; .02 current pinned resource YAML")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=4000)
    parser.add_argument("--bootstrap-time-limits", action="store_true",
                        help="Public StepCounter semantics: truncate at 100 and bootstrap value")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--evaluator-code", type=Path,
                        default=Path(os.environ.get("HYDROGYM_EVALUATOR_CODE", "/workspace/evaluation")))
    parser.add_argument("--evaluate-published", action="store_true")
    parser.add_argument("--record-probes", action="store_true",
                        help="Record raw u/v probe velocities on evaluation steps for physical fluctuation analysis")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.environment = "Cavity_2D_Re4200"
    if args.episodes <= 0 or args.steps <= 0 or (args.mode == "evaluate" and args.checkpoint is None):
        parser.error("Positive budgets and an evaluation checkpoint are required")
    if args.evaluate_published and args.mode != "train":
        parser.error("--evaluate-published is only used for training")
    if args.mode == "replay" and args.package is None:
        parser.error("A replay requires an exported policy package")
    for key in ("output", "data", "author_source", "restart", "restart_manifest"):
        setattr(args, key, getattr(args, key).resolve())
    for key in ("calibration", "checkpoint", "package", "evaluator_code"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    manifest = json.loads(args.restart_manifest.read_text())
    if (manifest["reynolds"] != 4200 or manifest["mach"] != .1
            or manifest["restart_sha256"] != digest(args.restart)):
        raise ValueError("Initial state differs from inspected native Re4200 restart")
    receipt = json.loads((args.data / "materialized.json").read_text())
    resources = args.data / "derived/Cavity_2D_Re4200"
    derivation = json.loads((resources / "derivation.json").read_text())
    if (receipt["revision"] != "cf5ef6748443ae72e9647f398487e98d421a7243"
            or derivation["parent_revision"] != receipt["revision"]):
        raise ValueError("Unverified cavity resources")
    for name, expected in derivation["generated_sha256"].items():
        if digest(resources / name) != expected:
            raise ValueError("Derived cavity properties changed")
    cached = Path.home() / ".cache/maiagym" / args.environment
    if cached.exists() and cached.resolve() != resources:
        raise ValueError("Legacy cache shadows the verified resource")
    import hydrogym.data_manager as manager
    manager.HF_AVAILABLE = False
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
        frozen = args.output / "source"
        frozen.mkdir()
        for name in ("cavity_train.py", "cavity_author.py", "cavity_reward.py", "public_learner.py", "time_limits.py"):
            shutil.copyfile(Path(__file__).with_name(name), frozen / name)
        shutil.copytree(Path(__file__).with_name("recipes"), frozen / "recipes")
        from hydrogym.maia.workspace import prepare_maia_workspace
        work, properties = prepare_maia_workspace(args.environment, work_dir=str(args.output / "cfd"),
            use_clean_cache=False, local_fallback_dir=str(resources.parent))
        private_restart = Path(work) / "out_lb/restart_.Netcdf"
        temporary = private_restart.with_name("initial-private.Netcdf")
        shutil.copyfile(args.restart, temporary)
        os.replace(temporary, private_restart)
        text, count = re.subn(r"(?m)^restartTimeStep\s*=.*$",
            f"restartTimeStep = {manifest['global_time_step']}", Path(properties).read_text())
        if count != 1:
            raise ValueError("Expected one restart time-step property")
        private_properties = args.output / "runtime-properties.toml"
        private_properties.write_text(text)
        command = ["mpirun", "--bind-to", "none", "-np", "1", sys.executable,
            str(frozen / "cavity_train.py"), *sys.argv[1:], "--worker", ":", "-np", "1",
            "maia", str(private_properties), "--silent"]
        write(args.output / "launch.json", {"command": command, "cwd": work,
            "restart": manifest, "runtime_properties_sha256": digest(private_properties)})
        environment = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                           OMPI_MCA_coll_hcoll_enable="0")
        subprocess.run(command, cwd=work, env=environment, check=True)
        if args.evaluate_published:
            hashes = set()
            for label, name in (("final", "candidate.pt"), ("best", "best.pt")):
                checkpoint = args.output / name
                sha = digest(checkpoint)
                if sha in hashes:
                    continue
                hashes.add(sha)
                command = [sys.executable, str(frozen / "cavity_train.py"), "evaluate",
                    "--data", str(args.data), "--author-source", str(args.author_source),
                    "--restart", str(args.restart), "--restart-manifest", str(args.restart_manifest),
                    "--output", str(args.output / f"independent-{label}"), "--algo", args.algo,
                    "--max-control", str(args.max_control),
                    "--seed", str(args.seed), "--checkpoint", str(checkpoint)]
                if args.calibration:
                    command.extend(["--calibration", str(args.calibration.resolve())])
                if args.bootstrap_time_limits:
                    command.append("--bootstrap-time-limits")
                if args.record_probes:
                    command.append("--record-probes")
                subprocess.run(command, env=environment, check=True)
            write(args.output / "workflow-completed.json", {"status": "cavity-trained-and-replayed-not-paper-matched",
                  "policy_sha256": sorted(hashes)})
        return

    import gymnasium as gym
    import hydrogym.maia as maia
    import numpy as np
    from cavity_author import load
    from cavity_reward import CavityReward, evaluate_with_diagnostics
    probes, vectors, author_audit = load(args.author_source)
    env = maia.from_hf(args.environment, use_clean_cache=False, local_fallback_dir=str(resources.parent),
        probe_locations=probes, obs_normalization_strategy="none")
    if Path(env.env_data_path).resolve() != resources:
        raise ValueError("Native worker used different resources")
    native_max_control = float(env.MAX_CONTROL)
    env.num_substeps_per_iteration, env.max_episode_steps, env.MAX_CONTROL = 400, 99, args.max_control
    if args.bootstrap_time_limits:
        env.max_episode_steps = 100
    env.obs_loc, env.obs_scale = vectors["obs_loc"], vectors["obs_scale"]
    reward = CavityReward(vectors["reward_loc"], vectors["reward_scale"])
    if args.calibration:
        reward = CavityReward.from_calibration(args.calibration, reynolds=4200,
            probe_locations=probes, observation_types=["u", "v"], substeps=400)
    last = {}

    def eq36():
        score, diagnostics = evaluate_with_diagnostics(reward, env.obs, record_probes=args.record_probes)
        last.clear()
        last.update(diagnostics)
        return score, {}

    env.get_reward = eq36
    targets = json.dumps([reward.mean.tolist(), reward.std.tolist()]).encode()
    protocol = {"status": "cavity-public-description-reconstruction-not-paper-matched",
        "environment": args.environment, "protocol": "cavity-author-reconstruction",
        "probe_locations": probes, "effective_max_control": args.max_control,
        "native_max_control": native_max_control,
        "control_limit_source": "public author YAML" if args.max_control == .075 else "pinned resource YAML",
        "effective_omega": float(env.omega),
        "effective_substeps": 400, "horizon": 100, "observation_normalization": "author-example",
        "time_limit_bootstrap": args.bootstrap_time_limits,
        "observation_loc": env.obs_loc.tolist(), "observation_scale": env.obs_scale.tolist(),
        "reward": "supplementary Eq.36", "reward_target_sha256": hashlib.sha256(targets).hexdigest(),
        "target_source": str(args.calibration) if args.calibration else "public author example",
        "initial_restart_sha256": manifest["restart_sha256"], "initial_state": manifest,
        "author_example": author_audit, "seed": args.seed,
        "recorded_probe_velocity": {"enabled": args.record_probes,
            "units": "lattice_velocity", "order": "all_u_then_all_v"},
        "unresolved": ["exact figure-specific initial state, target statistics and training recipe",
                       "100-action horizon is public-example default, not explicit figure metadata"]}
    write(args.output / "protocol.json", protocol)

    class Logged(gym.Wrapper):
        phase = "training"
        training_step = 0
        episode = 0

        def __init__(self):
            super().__init__(env)
            self.stream = (args.output / "episodes.jsonl").open("w", buffering=1)
            self.trajectory = (args.output / "evaluation-steps.jsonl").open("w", buffering=1)
            self.controls = (args.output / "controls.jsonl").open("w", buffering=1)

        def reset(self, **kwargs):
            self.episode += 1
            self.count, self.total, self.started = 0, 0., time.monotonic()
            obs, info = self.env.reset(**kwargs)
            with np.errstate(over="ignore", invalid="ignore"):
                obs = np.asarray(obs, dtype=np.float32)
            if not np.isfinite(obs).all():
                raise ValueError("Nonfinite native reset")
            return obs.astype(np.float32), info

        def step(self, action):
            action = np.asarray(action, dtype=self.action_space.dtype)
            if not np.isfinite(action).all() or not self.action_space.contains(action):
                raise ValueError("Invalid native action")
            self.controls.write(json.dumps({"episode": self.episode, "phase": self.phase,
                "step": self.count + 1, "training_step": self.training_step,
                "normalized_action": action.tolist()}, allow_nan=False) + "\n")
            try:
                obs, score, terminated, truncated, info = self.env.step(action)
            except Exception as error:
                raw = np.asarray(self.env.obs)
                finite = raw[np.isfinite(raw)]
                write(args.output / "transition-error.json", {
                    "error": repr(error), "episode": self.episode,
                    "control_in_episode": self.count + 1,
                    "training_step": self.training_step, "phase": self.phase,
                    "normalized_action": action.tolist(),
                    "physical_control": (action * args.max_control).tolist(),
                    "observation_shape": list(raw.shape),
                    "nonfinite_observations": int((~np.isfinite(raw)).sum()),
                    "max_abs_finite_observation": float(np.abs(finite).max()) if finite.size else None})
                raise
            with np.errstate(over="ignore", invalid="ignore"):
                obs = np.asarray(obs, dtype=np.float32)
            if not np.isfinite(obs).all() or not np.isfinite(score):
                raise ValueError("Nonfinite native transition")
            self.count += 1
            self.total += float(score)
            terminated, truncated = apply_time_limit(terminated, truncated, self.count,
                                                      100, args.bootstrap_time_limits)
            if bool(terminated or truncated) != (self.count == 100):
                raise ValueError("Unexpected native episode length")
            if self.phase == "evaluation":
                self.trajectory.write(json.dumps({"episode": self.episode, "step": self.count,
                    "training_step": self.training_step, "reward": float(score),
                    "normalized_action": action.tolist(), "physical_control": (action * args.max_control).tolist(),
                    **last}, allow_nan=False) + "\n")
            if terminated or truncated:
                record = {"episode": self.episode, "steps": self.count, "phase": self.phase,
                    "terminated": terminated, "truncated": truncated,
                    "training_step": self.training_step, "return": self.total,
                    "mean_reward": self.total / self.count, "wall_sec": time.monotonic() - self.started}
                self.stream.write(json.dumps(record) + "\n")
                print("EPISODE", json.dumps(record), flush=True)
            return obs.astype(np.float32), float(score), terminated, truncated, info

        def close(self):
            self.stream.close()
            self.trajectory.close()
            self.controls.close()
            self.env.close()

    wrapped = Logged()
    try:
        if args.mode == "zero":
            wrapped.phase = "evaluation"
            for _ in range(args.episodes):
                wrapped.reset(seed=args.seed)
                for _ in range(100):
                    wrapped.step(np.zeros(env.action_space.shape))
            write(args.output / "completed.json", {"status": "native-cavity-eq36-zero-replay"})
        elif args.mode == "replay":
            sys.path.insert(0, str(args.evaluator_code))
            from evaluate import package_hash
            from protocol import Policy, replay_episode
            before = package_hash(args.package)
            snapshot = args.output / "relocated-policy"
            shutil.copytree(args.package, snapshot)
            if package_hash(snapshot) != before:
                raise ValueError("Package changed while creating replay snapshot")
            wrapped.phase = "evaluation"
            policy = Policy(snapshot, args.output / "policy.stderr", timeout=120)
            try:
                case = {"environment": args.environment, "family": "cavity", "horizon": 100}
                rows = [replay_episode(wrapped, policy, case, args.seed + i) for i in range(args.episodes)]
                policy.close()
            finally:
                policy.abort()
            if package_hash(snapshot) != before:
                raise ValueError("Policy package mutated during replay")
            write(args.output / "completed.json", {"status": "relocated-native-cavity-replay-not-paper-matched",
                                                  "episodes": rows, "package_hash": before})
        else:
            # Zero control and portable policy serving do not need TorchRL.
            from public_learner import run
            run(wrapped, args, protocol)
    except BaseException:
        # Native reward exceptions occur during an unfinished MPI exchange.
        # finishRun sends the next timestep, while MAIA still waits for a data
        # command, so ordinary close can deadlock and conceal the traceback.
        # Preserve the error before aborting this job's paired ranks only.
        detail = traceback.format_exc()
        write(args.output / "worker-error.json", {"traceback": detail,
            "episode": wrapped.episode, "training_step": wrapped.training_step})
        print(detail, file=sys.stderr, flush=True)
        wrapped.stream.flush()
        wrapped.trajectory.flush()
        wrapped.controls.flush()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
        raise
    else:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if "--worker" in sys.argv:
            import traceback
            from mpi4py import MPI
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
