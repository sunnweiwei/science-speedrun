"""Run Supplementary Table S16 SAM-binding hydrogen-bond beam search cohorts."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import yaml

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "docs/s15"))
from run_cohort import prior_schedule

S16_SAMPLERS = ("stochastic-tsr", "prior-tsr", "full-sde", "short-sde")
MEDIUM_MIDPEAK = [[5, 1], [5, 2], [5, 2], [5, 4], [10, 5], [5, 4], [5, 2], [5, 2], [15, 1]]


def build_s16_cfg(
    arm: str,
    schedule_mode: str,
    sam_cif: Path,
    assets: Path,
    out_dir: Path,
    replicates: int,
    batch_size: int,
    seed: int,
) -> dict:
    if arm not in S16_SAMPLERS:
        raise ValueError(f"Expected arm in {S16_SAMPLERS}, got {arm!r}")
    nsteps = 400 if arm == "full-sde" else 60
    cp1, cp2 = (100, 200) if arm == "full-sde" else (15, 30)
    if schedule_mode == "base_1pass":
        bands = [[nsteps, 1]]
    elif schedule_mode == "midpeak":
        if arm == "full-sde":
            total = last = 0
            bands = []
            for w, d in MEDIUM_MIDPEAK:
                total += w
                b = round(400 * total / 60)
                bands.append([b - last, d])
                last = b
        else:
            bands = [list(pair) for pair in MEDIUM_MIDPEAK]
    else:
        raise ValueError(f"Unknown schedule_mode: {schedule_mode}")

    cfg = {
        "defaults": ["inference", "_self_"],
        "inputs": str(sam_cif),
        "out_dir": str(out_dir),
        "hf_ckpt_path": str(assets / "rfd4"),
        "compile_lengths": None,
        "num_workers": 0,
        "seed": seed,
        "batch_size": min(batch_size, replicates),
        "num_replicates": replicates,
        "skip_existing": False,
        "generation": {"args": {"nsteps": nsteps}, "n_recycle_schedule": bands},
        "metrics_tags": ["interchain_interactions"],
        "search": {
            "beam_width": 2,
            "n_branch": 2,
            "save_rollouts": False,
            "checkpoints": {
                str(cp1): {
                    "metrics": ["interchain_interactions"],
                    "rewards": {"interchain_interactions.hbond_atom_pair_count_coi": {"weight": 1.0}},
                },
                str(cp2): {
                    "metrics": ["interchain_interactions"],
                    "rewards": {"interchain_interactions.hbond_atom_pair_count_coi": {"weight": 1.0}},
                },
            },
        },
    }
    if arm in ("prior-tsr", "full-sde", "short-sde"):
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
    cfg = build_s16_cfg(
        args.arm, args.schedule_mode, args.sam_cif.resolve(), args.assets.resolve(),
        work / "outputs", args.replicates, args.batch_size, args.seed,
    )
    (work / "s16_cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

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
    sys.argv = ["rfd4-generate", "--config-dir", str(work), "--config-name", "s16_cfg"]
    main_hydra()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "_worker"), default="run")
    p.add_argument("--arm", choices=S16_SAMPLERS, required=True)
    p.add_argument("--schedule-mode", choices=("base_1pass", "midpeak"), default="base_1pass")
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=727000)
    p.add_argument("--sam-cif", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    if args.stage == "_worker":
        return run_worker(args)

    work = args.work.resolve()
    done_file = work / "done.json"
    if done_file.is_file():
        print(f"Skipping completed S16 cohort {work}")
        return
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--stage", "_worker",
        "--arm", args.arm,
        "--schedule-mode", args.schedule_mode,
        "--replicates", str(args.replicates),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--sam-cif", str(args.sam_cif.resolve()),
        "--assets", str(args.assets.resolve()),
        "--work", str(work),
    ]
    env = dict(os.environ, DATA_PATH=str(args.assets.resolve()), HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    if args.project_root is not None:
        env["PROJECT_ROOT"] = str(args.project_root.resolve())

    t0 = time.monotonic()
    with (work / "generation.log").open("w") as f:
        subprocess.run(cmd, env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True)
    wall_s = time.monotonic() - t0

    with (work / "outputs/inference_metrics/per_example_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    expected_designs = args.replicates * 2
    if len(rows) != expected_designs:
        raise RuntimeError(f"Expected {expected_designs} S16 survivors, got {len(rows)}")
    hbonds = [float(r["interchain_interactions.hbond_atom_pair_count_coi"]) for r in rows]
    batches = [json.loads(line) for line in (work / "sampling.jsonl").read_text().splitlines()]
    warm_batches = batches[1:] if len(batches) > 1 else batches
    warm_s_per_run = sum(b["seconds"] for b in warm_batches) / sum(b["size"] for b in warm_batches)

    summary = {
        "arm": args.arm,
        "schedule_mode": args.schedule_mode,
        "search_runs": args.replicates,
        "final_designs": len(hbonds),
        "mean_hbond_count": statistics.fmean(hbonds),
        "stdev_hbond_count": statistics.stdev(hbonds) if len(hbonds) > 1 else 0.0,
        "median_hbond_count": statistics.median(hbonds),
        "min_hbond_count": min(hbonds),
        "max_hbond_count": max(hbonds),
        "wall_seconds": wall_s,
        "warm_s_per_search_run": warm_s_per_run,
        "hbonds": hbonds,
    }
    done_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
