"""Run length-only unconditional monomer cohorts across base (n_recycle=0), thinking, and multi-seed settings."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "environment/evaluation"))
sys.path.insert(0, str(TASK / "docs/s15"))
from policy import LENGTHS
from run_cohort import MONOMER_HASHES, SAMPLERS, prior_schedule

THINKING_BANDS = {
    "base_60": [[60, 1]],
    "base_400": [[400, 1]],
    "low": [[10, 4], [10, 3], [10, 2], [30, 1]],
    "medium": [[5, 6], [10, 5], [5, 4], [10, 3], [20, 2], [10, 1]],
    "high": [[10, 9], [10, 8], [10, 5], [20, 3], [10, 2]],
}


def build_monomer_cfg(
    arm: str,
    schedule_mode: str,
    length: int,
    slots: int,
    gen_seed: int,
    assets: Path,
    out_dir: Path,
    batch_size: int = 5,
) -> dict:
    if arm not in SAMPLERS or length not in LENGTHS:
        raise ValueError(f"Expected sampler in {SAMPLERS} and length in {tuple(LENGTHS)}")
    if schedule_mode == "base_1pass":
        bands = THINKING_BANDS["base_400"] if arm == "full-sde" else THINKING_BANDS["base_60"]
        nsteps = 400 if arm == "full-sde" else 60
    elif schedule_mode == "release_medium":
        if arm == "full-sde":
            total = last = 0
            bands = []
            for w, d in THINKING_BANDS["medium"]:
                total += w
                b = round(400 * total / 60)
                bands.append([b - last, d])
                last = b
            nsteps = 400
        else:
            bands = [list(pair) for pair in THINKING_BANDS["medium"]]
            nsteps = 60
    elif schedule_mode in ("low", "medium", "high"):
        if arm == "full-sde":
            raise ValueError(f"schedule_mode={schedule_mode!r} is defined for 60-step arms")
        bands = [list(pair) for pair in THINKING_BANDS[schedule_mode]]
        nsteps = 60
    else:
        raise ValueError(f"Unknown schedule_mode: {schedule_mode}")

    cfg = {
        "defaults": ["inference", "_self_"],
        "inputs": str(assets / "monomers" / f"monomer-{LENGTHS[length]}.cif.zst"),
        "out_dir": str(out_dir),
        "hf_ckpt_path": str(assets / "rfd4"),
        "compile_lengths": None,
        "num_workers": 0,
        "seed": gen_seed,
        "batch_size": min(batch_size, slots),
        "num_replicates": slots,
        "skip_existing": False,
        "generation": {"args": {"nsteps": nsteps}, "n_recycle_schedule": bands},
        "metrics_tags": [],
    }
    if arm == "ode-tsr":
        cfg["generation"]["model"] = {m: {"gt": {"mode": "ode"}} for m in ("bb_ca", "local_latents")}
    elif arm in ("prior-tsr", "full-sde", "short-sde"):
        mode = "vf_tsr" if arm == "prior-tsr" else "sc"
        cfg["generation"]["model"] = {
            modality: {"simulation_step_params": {"sampling_mode": mode, "sc_scale_noise": variance}}
            for modality, variance in (("bb_ca", 0.3), ("local_latents", 0.1))
        }
    return cfg


def run_worker(args):
    import torch
    from rfproteina.generate import main_hydra
    from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
    from rfproteina.proteina import Proteina

    work = args.work.resolve()
    assets = args.assets.resolve()
    cfg = build_monomer_cfg(
        args.arm, args.schedule_mode, args.length, args.slots,
        args.gen_seed, assets, work / "outputs", args.batch_size,
    )
    if hashlib.sha256(Path(cfg["inputs"]).read_bytes()).hexdigest() != MONOMER_HASHES[args.length]:
        raise ValueError("Unexpected monomer input digest")
    (work / "run_cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if args.arm == "prior-tsr":
        native_schedule = ProductSpaceFlowMatcher.sample_schedule

        def unscaled_beta(self, nsteps, sampling_model_args):
            return prior_schedule(native_schedule, self, nsteps, sampling_model_args)

        ProductSpaceFlowMatcher.sample_schedule = unscaled_beta

    orig_predict = Proteina.predict_step

    def timed_predict(self, batch, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        res = orig_predict(self, batch, *a, **kw)
        torch.cuda.synchronize()
        with (work / "sampling.jsonl").open("a") as f:
            f.write(json.dumps({"size": int(batch["padding_mask"].shape[0]), "seconds": time.monotonic() - t0}) + "\n")
        return res

    Proteina.predict_step = timed_predict
    sys.argv = ["rfd4-generate", "--config-dir", str(work), "--config-name", "run_cfg", "~metrics"]
    main_hydra()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "_worker"), default="run")
    p.add_argument("--arm", choices=SAMPLERS, required=True)
    p.add_argument("--schedule-mode", choices=("base_1pass", "release_medium", "low", "medium", "high"), required=True)
    p.add_argument("--length", type=int, choices=tuple(LENGTHS), required=True)
    p.add_argument("--slots", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--gen-seed", type=int, required=True)
    p.add_argument("--af3-seed", type=int, default=727000)
    p.add_argument("--extra-af3-seed", type=int, default=None)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    if args.stage == "_worker":
        return run_worker(args)

    work = args.work.resolve()
    done_file = work / "done.json"
    if done_file.is_file():
        print(f"Skipping completed cohort {work}")
        return
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--stage", "_worker",
        "--arm", args.arm,
        "--schedule-mode", args.schedule_mode,
        "--length", str(args.length),
        "--slots", str(args.slots),
        "--batch-size", str(args.batch_size),
        "--gen-seed", str(args.gen_seed),
        "--assets", str(args.assets.resolve()),
        "--work", str(work),
    ]
    env = dict(os.environ, DATA_PATH=str(args.assets.resolve()), HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    t0 = time.monotonic()
    with (work / "generation.log").open("w") as f:
        subprocess.run(cmd, env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True)
    gen_wall_s = time.monotonic() - t0

    with (work / "outputs/inference_metrics/per_example_metrics.csv").open() as f:
        paths = sorted(r["designed_structure_path"] for r in csv.DictReader(f))
    if len(paths) != args.slots or len(set(paths)) != args.slots or not all(Path(q).is_file() for q in paths):
        raise RuntimeError(f"Incomplete native output manifest in {work}")
    batches = [json.loads(line) for line in (work / "sampling.jsonl").read_text().splitlines()]
    if sum(b["size"] for b in batches) != args.slots:
        raise RuntimeError(f"Incomplete sampling timer in {work}")

    cfg = yaml.safe_load((work / "run_cfg.yaml").read_text())
    passes_per_slot = sum(w * d for w, d in cfg["generation"]["n_recycle_schedule"])
    warm_batches = batches[1:] if len(batches) > 1 else batches
    warm_s_per_design = sum(b["seconds"] for b in warm_batches) / sum(b["size"] for b in warm_batches)

    gen_receipt = {
        "arm": args.arm,
        "schedule_mode": args.schedule_mode,
        "length": args.length,
        "gen_seed": args.gen_seed,
        "slots": args.slots,
        "batch_size": args.batch_size,
        "trunk_passes_per_slot": passes_per_slot,
        "generation_seconds": gen_wall_s,
        "warm_predict_s_per_design": warm_s_per_design,
        "paths": paths,
        "batches": batches,
    }
    (work / "generation.json").write_text(json.dumps(gen_receipt, indent=2) + "\n")

    from score import score

    scored = score({args.length: [Path(q) for q in paths]}, args.slots, args.assets.resolve(), work / "evaluation", args.af3_seed)
    (work / "result.json").write_text(json.dumps(scored, indent=2, allow_nan=False) + "\n")

    extra_scored = None
    if args.extra_af3_seed is not None:
        extra_scored = score(
            {args.length: [Path(q) for q in paths]}, args.slots, args.assets.resolve(),
            work / f"evaluation_seed_{args.extra_af3_seed}", args.extra_af3_seed,
        )
        (work / f"result_seed_{args.extra_af3_seed}.json").write_text(
            json.dumps(extra_scored, indent=2, allow_nan=False) + "\n"
        )

    done_payload = {
        "generation": {k: v for k, v in gen_receipt.items() if k not in ("paths", "batches")},
        "af3_seed": args.af3_seed,
        "score": scored,
        "extra_af3_seed": args.extra_af3_seed,
        "extra_score": extra_scored,
    }
    done_file.write_text(json.dumps(done_payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
