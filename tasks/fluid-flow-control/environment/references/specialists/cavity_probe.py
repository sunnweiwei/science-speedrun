"""Uncontrolled native cavity measurements for public-protocol reconstruction.

This is an organizer diagnostic, not a scored policy or the paper's 1,000-cycle
target calibration. It retains public Re4140 geometry/restart without silently
renaming it Re4200. Probe locations are explicitly reconstructed for that mesh.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--environment", choices=["Cavity_2D_Re4140", "Cavity_2D_Re4200"], default="Cavity_2D_Re4140")
    parser.add_argument("--min-calibration-cycles", type=int, default=0)
    parser.add_argument("--restart-interval", type=int, default=0,
                        help="Explicit solver restart-output interval; uses private writable input copy")
    parser.add_argument("--probe-layout", choices=["reconstructed25", "author34"], default="reconstructed25")
    parser.add_argument("--author-source", type=Path, default=Path(os.environ.get("HYDROGYM_AUTHOR_SOURCE", "/author-source")))
    parser.add_argument("--reward", choices=["native", "author-eq36"], default="native")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.steps < 200:
        parser.error("Use at least 200 samples for this diagnostic")
    if args.restart_interval < 0:
        parser.error("restart-interval must be nonnegative")
    if args.reward == "author-eq36" and args.probe_layout != "author34":
        parser.error("Author Eq.36 vectors are only defined for the author34 probe order")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    name = args.environment
    receipt = json.loads((args.data / "materialized.json").read_text())
    if receipt["revision"] != "cf5ef6748443ae72e9647f398487e98d421a7243" or "Cavity_2D_Re4140" not in receipt["verified_directories"]:
        raise RuntimeError("Pinned cavity resources have not been verified")
    import hydrogym.data_manager as manager
    manager.HF_AVAILABLE = False
    source = args.data / "environments" / name
    derivation = None
    if name == "Cavity_2D_Re4200":
        source = args.data / "derived" / name
        derivation = json.loads((source / "derivation.json").read_text())
        if derivation["parent_revision"] != receipt["revision"] or derivation["parent_environment"] != "Cavity_2D_Re4140":
            raise RuntimeError("Unexpected derived resource parent")
        for filename, expected in derivation["generated_sha256"].items():
            if hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Derived resource changed after materialization")
    cached = Path.home() / ".cache/maiagym" / name
    if cached.exists() and cached.resolve() != source:
        raise RuntimeError("Legacy cache shadows pinned resources")
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(__file__, args.output / "source.py")
        if args.probe_layout == "author34":
            for module in ("cavity_author.py", "cavity_reward.py"):
                shutil.copyfile(Path(__file__).with_name(module), args.output / module)
        from hydrogym.maia.workspace import prepare_maia_workspace
        work, properties = prepare_maia_workspace(name, work_dir=str(args.output / "cfd"),
            use_clean_cache=False, local_fallback_dir=str(source.parent))
        if args.restart_interval:
            # Never let a solver writer follow a link into original resources.
            # Keep a private input even if this MAIA build writes restart_.Netcdf.
            restart = Path(work) / "out_lb/restart_.Netcdf"
            temporary = restart.with_name("restart-private.Netcdf")
            shutil.copyfile(restart, temporary)
            os.replace(temporary, restart)
            text, count = re.subn(r"(?m)^restartInterval\s*=.*$",
                                 f"restartInterval = {args.restart_interval}",
                                 Path(properties).read_text())
            if count != 1:
                raise RuntimeError("Expected one restartInterval property")
            private = args.output / "restart-output-properties.toml"
            private.write_text(text)
            properties = str(private)
        command = ["mpirun", "--bind-to", "none", "-np", "1", sys.executable,
                   str(args.output / "source.py"), "--worker", "--data", str(args.data),
                   "--output", str(args.output), "--steps", str(args.steps),
                   "--environment", name, "--min-calibration-cycles", str(args.min_calibration_cycles),
                   "--restart-interval", str(args.restart_interval),
                   "--probe-layout", args.probe_layout, "--author-source", str(args.author_source),
                   "--reward", args.reward,
                   ":", "-np", "1", "maia", properties, "--silent"]
        write(args.output / "command.json", command)
        subprocess.run(command, cwd=work, check=True, env=dict(os.environ,
            OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
            OMPI_MCA_coll_hcoll_enable="0"))
        return

    import numpy as np
    import hydrogym.maia as maia
    from scipy.signal import periodogram, find_peaks

    # Public geometry: cavity x=[1.2,2.2], y=[-1,0]. Keep all probes in fluid.
    # Old author defaults x=26..42 do not fit this mesh and cannot be copied.
    probes = [float(c) for x in np.linspace(1.25, 2.15, 5)
              for y in np.linspace(-0.5, 0.1, 5) for c in (x, y)]
    author_audit = None
    if args.probe_layout == "author34":
        from cavity_author import load
        probes, vectors, author_audit = load(args.author_source)
    env = maia.from_hf(name, use_clean_cache=False, local_fallback_dir=str(source.parent),
                       probe_locations=probes, obs_normalization_strategy="none")
    if Path(env.env_data_path).resolve() != source:
        raise RuntimeError("Native environment loaded unexpected resources")
    # Public cavity YAML omits maia.U_inf, required by the native U_inf option.
    # Derive the lattice freestream velocity from the actual solver Mach number,
    # using the same c_s=1/sqrt(3) relation as native force normalization.
    env.obs_loc = np.zeros(env.num_outputs)
    env.obs_scale = np.full(env.num_outputs, float(env.Ma) / np.sqrt(3))
    env.num_substeps_per_iteration = 400  # Supplementary section 5C.
    env.max_episode_steps = args.steps - 1  # A continuous diagnostic, not training episodes.
    env._ensure_baseline_loaded()
    if args.reward == "author-eq36":
        from cavity_reward import CavityReward
        eq36 = CavityReward(vectors["reward_loc"], vectors["reward_scale"])
        env.get_reward = lambda: (eq36(env.obs), {})
        env.obs_loc = vectors["obs_loc"]
        env.obs_scale = vectors["obs_scale"]
    protocol = {"status": "uncontrolled-cavity-diagnostic-not-paper-calibration",
        "environment": name, "source_revision": receipt["revision"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reynolds": float(env.Re), "substeps_per_action": 400, "steps": args.steps,
        "derivation": derivation, "minimum_calibration_cycles": args.min_calibration_cycles,
        "restart_output_interval": args.restart_interval,
        "probe_layout": args.probe_layout, "author_example": author_audit,
        "action": "zero", "probe_locations": probes,
        "probe_provenance": "rescaled-public-author-example" if author_audit else "organizer-reconstruction",
        "observations": list(env.observation_type), "observation_scale": env.obs_scale.tolist(),
        "normalization_reconstruction": ("public author-example observation vectors" if args.reward == "author-eq36"
            else "lattice U_inf=Ma/sqrt(3); missing native YAML maia.U_inf"),
        "published_target_mean_sha256": hashlib.sha256((source / "baseline_state.feather").read_bytes()).hexdigest(),
        "reward": ("Eq.36 with unchanged public author-example target vectors" if args.reward == "author-eq36"
                   else "unchanged native absolute standardized deviation (not paper Eq.36)"),
        "unresolved": ["Re4140 resource versus paper Re4200", "exact figure probe coordinates",
                       "paper 1000-instability-cycle target statistics", "geometry/refinement discrepancies"]}
    write(args.output / "protocol.json", protocol)
    raw_observations = []
    native_reward = env.get_reward
    with open(os.devnull, "w") as quiet:
        def recorded_reward():
            raw_observations.append(np.asarray(env.obs).copy())
            # Suppress only the native debug vector print, not solver errors.
            with contextlib.redirect_stdout(quiet):
                return native_reward()
        env.get_reward = recorded_reward
        started = time.monotonic()
        records = []
        try:
            obs, _ = env.reset(seed=0)
            if not np.isfinite(obs).all():
                raise RuntimeError("Nonfinite cavity reset")
            with (args.output / "observations.jsonl").open("w", buffering=1) as stream:
                for step in range(args.steps):
                    obs, reward, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
                    raw = raw_observations[-1]
                    if not np.isfinite(raw).all() or not np.isfinite(reward):
                        raise RuntimeError(f"Nonfinite native cavity trajectory at {step + 1}")
                    record = {"step": step + 1, "native_reward": float(reward), "raw_observation": raw.tolist()}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(float(reward))
                    if bool(terminated or truncated) != (step + 1 == args.steps):
                        raise RuntimeError("Unexpected cavity diagnostic termination")
                    if (step + 1) % 100 == 0:
                        print(json.dumps({"step": step + 1, "wall_seconds": time.monotonic() - started,
                                          "native_reward": float(reward)}), flush=True)
        finally:
            env.close()

    samples = np.asarray(raw_observations)
    np.savez_compressed(args.output / "raw-measurements.npz", observations=samples,
                        native_rewards=np.asarray(records), published_mean=env.baseline_values)
    # Diagnose oscillation period using the most variable v probe, not a claimed
    # paper instability frequency. Record both spectral and peak-count estimates.
    tail = samples[len(samples) // 2:]
    v_index = tail.shape[1] // 2 + int(np.argmax(tail[:, tail.shape[1] // 2:].var(axis=0)))
    signal = tail[:, v_index]
    frequencies, power = periodogram(signal, detrend="linear")
    dominant = int(np.argmax(power[1:])) + 1
    period = 1 / frequencies[dominant]
    peaks, _ = find_peaks(signal, prominence=float(signal.std()) * 0.5)
    result = {"status": "uncontrolled-native-cavity-measurements-not-paper-calibration",
        "samples": len(samples), "tail_samples": len(tail), "wall_seconds": time.monotonic() - started,
        "mean": tail.mean(axis=0).tolist(), "std_population": tail.std(axis=0).tolist(),
        "nonzero_std_all_probes": bool((tail.std(axis=0) > 0).all()),
        "period_probe_index": v_index, "dominant_period_control_steps": float(period),
        "tail_peak_count": len(peaks), "calibrated_1000_cycles": False,
        "native_mean_reward": float(np.mean(records)),
        "published_mean_max_abs_difference": float(np.abs(tail.mean(axis=0) - env.baseline_values).max())}
    if args.min_calibration_cycles:
        target_cycles = args.min_calibration_cycles
        covered = len(tail) / period >= target_cycles and len(peaks) >= target_cycles
        chunks = np.array_split(tail, 4)
        max_drift = float(max(np.max(np.abs(chunk.mean(axis=0) - tail.mean(axis=0)) /
                                      np.maximum(tail.std(axis=0), 1e-15)) for chunk in chunks))
        result.update(status="reconstructed-cavity-target-statistics-not-paper-certified",
                      requested_cycles=target_cycles, estimated_tail_cycles=float(len(tail) / period),
                      estimated_cycle_coverage_met=bool(covered), max_block_mean_drift_in_std=max_drift)
        # Keep the field false: this is an inferred-period reconstruction, not
        # verification of the undisclosed original target/calibration protocol.
        if not covered or not result["nonzero_std_all_probes"]:
            result["status"] = "insufficient-cavity-target-calibration"
    write(args.output / "diagnostic.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if "--worker" in sys.argv:
            # A failed Python MPMD worker must not leave its paired solver
            # waiting indefinitely for a control message.
            import traceback
            from mpi4py import MPI
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
