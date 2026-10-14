"""Create a randomly initialized actor package (no torch): the untrained anchor and a plumbing check."""

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from tokagym import CASES, TokamakMagneticEnv  # noqa: E402
from common import N_OBS, Scaler, make_scaler, save_actor  # noqa: E402
from publish import publish  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-dir", required=True)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--hidden", nargs="*", type=int, default=[256, 256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=0.01, help="output layer scale; small = near-zero volts")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    env = TokamakMagneticEnv(sorted(CASES)[0], randomization="none")
    scaler = Scaler(make_scaler(env.probe_scale))
    sizes = [N_OBS] + args.hidden + [12]
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = rng.normal(0, np.sqrt(2.0 / a), (a, b)) * (args.gain if i == len(sizes) - 2 else 1.0)
        params.append((W, np.zeros(b)))
    save_actor(args.actor_dir, params, scaler, env.vmax, 1.0,
               {"env_version": "tokagym-mastu/v0.2", "algorithm": "random-init"})
    print("actor saved to", args.actor_dir)
    if args.submission:
        print("published to", publish(args.actor_dir, args.submission, {"random_init": True}))


if __name__ == "__main__":
    main()
