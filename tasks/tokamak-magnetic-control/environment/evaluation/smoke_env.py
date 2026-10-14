"""Smoke check: build every case, run a few zero-voltage steps, print timings and metrics."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokagym import CASES, TokamakMagneticEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=sorted(CASES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--solver", default="nonlinear")
    ap.add_argument("--randomization", default="visible")
    ap.add_argument("--policy", default="steady", choices=["steady", "zero", "random"])
    args = ap.parse_args()
    for name in args.cases:
        t0 = time.monotonic()
        env = TokamakMagneticEnv(name, randomization=args.randomization, solver=args.solver)
        print(f"[{name}] reference built in {time.monotonic()-t0:.1f}s; n_obs={env.n_obs} "
              f"targets={len(env.target_points)} xpts={len(env.target_xpoints)} Ip={env.reference_ip:.0f}",
              flush=True)
        obs, info = env.reset(seed=1)
        print(f"[{name}] reset {info['setup_time_sec']:.1f}s delay={info['supply_delay_steps']} "
              f"resistivity={info['plasma_resistivity']:.2e} params={info['profile_params']}", flush=True)
        steady = env._coil_currents() * np.asarray(env.tokamak.coil_resist[: env.n_active])
        rng = np.random.default_rng(0)
        for k in range(args.steps):
            if args.policy == "steady":
                a = steady
            elif args.policy == "zero":
                a = np.zeros(env.n_active)
            else:
                a = rng.uniform(-1, 1, env.n_active) * env.vmax * 0.2
            obs, r, term, trunc, info = env.step(a)
            print(f"  k={k+1} t={info['t']*1e3:.1f}ms step={info['step_time_sec']*1e3:.0f}ms "
                  f"shape_rms={info['shape_rms']*100:.2f}cm xerr={info['xpoint_error']*100:.2f}cm "
                  f"ip_err={info['ip_error_rel']*100:.2f}% Z={info['Z_axis']:.3f} viol={info['coil_violation']:.0f} "
                  f"r={r:.2f} term={term} obs_finite={np.isfinite(obs).all()}", flush=True)
            if term or trunc:
                print("  terminated:", info["termination"]); break
        print(f"[{name}] summary: {env.episode_summary()}", flush=True)


if __name__ == "__main__":
    main()
