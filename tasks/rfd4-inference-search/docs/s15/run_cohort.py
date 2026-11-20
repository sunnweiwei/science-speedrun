"""Run a fixed, length-only S15 cohort outside the 180-pass policy leaderboard."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "environment/evaluation"))
from policy import LENGTHS, native_config

SAMPLERS = ("stochastic-tsr", "prior-tsr", "ode-tsr", "full-sde", "short-sde")
MONOMER_HASHES = {100: "5e66073220a3314d3426ca5d7828c02492e61d24321966d4d73dff08481cfa37",
                  150: "06cb8093bcb51c8a923d352041db8bb9941b74236d0b2357d8c477204135ff50",
                  250: "26f764fa13f44288ca6214db3312505d646f2017a21e8329a40604731bd944bd"}
SEED = 727000
SLOTS = 50


def config(arm, length, assets, work):
    if arm not in SAMPLERS or length not in LENGTHS:
        raise ValueError("Expected a fixed S15 sampler and published monomer length")
    plan = json.loads((TASK / "environment/baseline/stochastic-tsr/policy.json").read_text())["default"]
    if arm == "ode-tsr":
        plan["solver"] = "ode-tsr"
    cfg = native_config(plan, length, SLOTS, SEED + length, assets, work / "outputs")
    if arm == "full-sde":
        # Scale cumulative boundaries of the 60-step released schedule to 400 steps.
        # Rounding the boundaries (rather than each width) preserves exactly 400 steps.
        total = 0
        last = 0
        bands = []
        for width, depth in plan["bands"]:
            total += width
            boundary = round(400 * total / 60)
            bands.append([boundary - last, depth])
            last = boundary
        cfg["generation"].update(n_recycle_schedule=bands, args={"nsteps": 400})
    if arm in ("prior-tsr", "full-sde", "short-sde"):
        mode = "vf_tsr" if arm == "prior-tsr" else "sc"
        cfg["generation"]["model"] = {
            modality: {"simulation_step_params": {"sampling_mode": mode, "sc_scale_noise": variance}}
            for modality, variance in (("bb_ca", 0.3), ("local_latents", 0.1))
        }
    return cfg


def prior_schedule(native_schedule, matcher, nsteps, args):
    from rfproteina.flow_matching.product_space_flow_matcher import get_gt

    # Keep the released TSR-safe time mesh; changing the solver mode here also changes that mesh.
    ts, _ = native_schedule(matcher, nsteps, args)
    beta = {}
    for modality, times in ts.items():
        curve = args[modality]["gt"]
        beta[modality] = get_gt(t=times[:-1], mode=curve["mode"], param=curve["p"], clamp_val=curve["clamp_val"])
    return ts, beta


def worker(arm, length, assets, work):
    import torch
    from rfproteina.generate import main_hydra
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    from rfproteina.proteina import Proteina

    cfg = config(arm, length, assets, work)
    if hashlib.sha256(Path(cfg["inputs"]).read_bytes()).hexdigest() != MONOMER_HASHES[length]:
        raise ValueError("Unexpected monomer input")
    (work / "s15_run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if arm == "prior-tsr":
        native_schedule = ProductSpaceFlowMatcher.sample_schedule

        def unscaled_beta(self, nsteps, sampling_model_args):
            return prior_schedule(native_schedule, self, nsteps, sampling_model_args)

        ProductSpaceFlowMatcher.sample_schedule = unscaled_beta

    original_predict = Proteina.predict_step

    def timed_predict(self, batch, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = original_predict(self, batch, *args, **kwargs)
        torch.cuda.synchronize()
        with (work / "sampling.jsonl").open("a") as f:
            f.write(json.dumps({"size": batch["padding_mask"].shape[0], "seconds": time.monotonic() - start}) + "\n")
        return result

    Proteina.predict_step = timed_predict
    sys.argv = ["rfd4-generate", "--config-dir", str(work), "--config-name", "s15_run", "~metrics"]
    main_hydra()


def cohort(args):
    assets = args.assets.resolve()
    work = args.root.resolve() / args.sampler / str(args.length)
    if args.stage == "_worker":
        return worker(args.sampler, args.length, assets, work)
    if args.stage == "generate":
        work.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, str(Path(__file__).resolve()), args.sampler, str(args.length), "_worker",
               "--root", str(args.root.resolve()), "--assets", str(assets)]
        env = dict(os.environ, DATA_PATH=str(assets), HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
        t0 = time.monotonic()
        with (work / "generation.log").open("w") as f:
            subprocess.run(cmd, env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True)
        seconds = time.monotonic() - t0
        with (work / "outputs/inference_metrics/per_example_metrics.csv").open() as f:
            paths = sorted(r["designed_structure_path"] for r in csv.DictReader(f))
        if len(paths) != SLOTS or len(set(paths)) != SLOTS or not all(Path(p).is_file() for p in paths):
            raise RuntimeError("Incomplete native output manifest")
        timings = [json.loads(line) for line in (work / "sampling.jsonl").read_text().splitlines()]
        if sum(row["size"] for row in timings) != SLOTS:
            raise RuntimeError("Incomplete sampling timer")
        receipt = {"sampler": args.sampler, "length": args.length, "seed": SEED + args.length,
                   "slots": SLOTS, "generation_seconds": seconds, "paths": paths, "batches": timings}
        (work / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        from score import score
        receipt = json.loads((work / "generation.json").read_text())
        report = score({args.length: [Path(p) for p in receipt["paths"]]}, SLOTS, assets, work / "evaluation", SEED)
        (work / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sampler", choices=SAMPLERS)
    p.add_argument("length", choices=tuple(LENGTHS), type=int)
    p.add_argument("stage", choices=("generate", "score", "_worker"))
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    cohort(p.parse_args())
