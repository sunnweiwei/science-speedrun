"""Run the official AF3-guided unconditional monomer beam across L=100, 150, 250."""
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

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "environment/evaluation"))
sys.path.insert(0, str(TASK / "docs/s15"))
from policy import LENGTHS
from run_cohort import MONOMER_HASHES


def build_hydra_argv(inp: Path, out_dir: Path, assets: Path, replicates: int, gen_seed: int, internal_oracle_seed: int) -> list[str]:
    return [
        "rfd4-generate",
        "inference_experiment=inference_beam_example",
        f"inputs={inp}",
        "subset_to_examples=null",
        f"out_dir={out_dir}",
        f"hf_ckpt_path={assets / 'rfd4'}",
        f"num_replicates={replicates}",
        "batch_size=1",
        f"seed={gen_seed}",
        "num_workers=0",
        "compile_lengths=null",
        "skip_existing=false",
        "search.save_rollouts=false",
        "search.save_criteria=null",
        f"++metrics.refolding_oracles.oracle.seed={internal_oracle_seed}",
    ]


def run_worker(args):
    import numpy as np
    import torch
    import rfproteina
    from rfproteina.generate import main_hydra
    from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
    from rfproteina.proteina import Proteina
    from rfproteina.search.beam_search import BeamSearchStrategy

    work = args.work.resolve()
    assets = args.assets.resolve()
    oracle_exec = str(args.oracle_exec)
    inp = assets / "monomers" / f"monomer-{LENGTHS[args.length]}.cif.zst"
    if hashlib.sha256(inp.read_bytes()).hexdigest() != MONOMER_HASHES[args.length]:
        raise ValueError("Unexpected monomer input digest")

    jax_cache = Path(os.environ.get("TMPDIR", str(work / "scratch"))) / "jax-cache-beam"
    jax_cache.mkdir(parents=True, exist_ok=True)
    worker_proc = None
    worker_log = (work / "af3_worker.log").open("w")
    oracle_calls = []
    checkpoint_rounds = []

    def wait_status():
        while True:
            line = worker_proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Persistent AF3 worker exited with {worker_proc.poll()}")
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "status" in msg:
                return msg

    def dispatch(oracle, manifest_path, daemon_rank, n_designs):
        nonlocal worker_proc
        if worker_proc is None:
            entry = Path(rfproteina.__file__).parent / "af3/run_alphafold_batch.py"
            cmd = [
                oracle_exec, str(entry), "--serve",
                "--num_recycles=5", "--num_diffusion_samples=1", "--buckets=256",
                "--flash_attention_implementation=xla", "--gpu_device=0",
                f"--model_dir={assets / 'af3'}",
                f"--jax_compilation_cache_dir={jax_cache}",
                "--save_full_outputs=false",
            ]
            worker_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True, bufsize=1,
            )
            if wait_status()["status"] != "ready":
                raise RuntimeError("Persistent AF3 worker failed startup")
        t0 = time.monotonic()
        worker_proc.stdin.write(str(Path(manifest_path).resolve()) + "\n")
        worker_proc.stdin.flush()
        reply = wait_status()
        if reply.get("status") != "ok" or reply.get("n_ok") != n_designs:
            raise RuntimeError(f"Internal AF3 beam refolding failed: {reply}")
        oracle_calls.append({"n_designs": n_designs, "elapsed_s": time.monotonic() - t0})

    orig_sim = BeamSearchStrategy._simulate_and_compute_reward

    def observe(self, model, batch, xt, current_step, reward_model, **kw):
        values, atoms, metrics = orig_sim(self, model, batch, xt, current_step, reward_model, **kw)
        if reward_model is not None:
            raw = [float(m["oracle.plddt_coi"]) for m in metrics]
            scored = values.detach().cpu().tolist()
            if not np.isfinite(raw).all() or not np.allclose(scored, np.maximum(raw, 50)):
                raise RuntimeError("Invalid internal beam pLDDT scores")
            checkpoint_rounds.append({"step": current_step, "plddt": raw, "reward": scored})
        return values, atoms, metrics

    orig_predict = Proteina.predict_step

    def timed_predict(self, batch, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        res = orig_predict(self, batch, *a, **kw)
        torch.cuda.synchronize()
        with (work / "sampling.jsonl").open("a") as f:
            f.write(json.dumps({"size": int(batch["padding_mask"].shape[0]), "seconds": time.monotonic() - t0}) + "\n")
        return res

    AF3Oracle._dispatch_fold = dispatch
    BeamSearchStrategy._simulate_and_compute_reward = observe
    Proteina.predict_step = timed_predict
    sys.argv = build_hydra_argv(inp, work / "outputs", assets, args.replicates, args.gen_seed, args.internal_oracle_seed)
    try:
        main_hydra()
    finally:
        (work / "beam_trace.json").write_text(
            json.dumps({"oracle_calls": oracle_calls, "checkpoints": checkpoint_rounds}, indent=2, allow_nan=False) + "\n"
        )
        if worker_proc is not None:
            try:
                worker_proc.stdin.write("shutdown\n")
                worker_proc.stdin.flush()
                worker_proc.wait(timeout=12)
            except Exception:
                worker_proc.kill()
                worker_proc.wait()
        worker_log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "_worker"), default="run")
    p.add_argument("--length", type=int, choices=tuple(LENGTHS), required=True)
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--gen-seed", type=int, required=True)
    p.add_argument("--internal-oracle-seed", type=int, default=727000)
    p.add_argument("--af3-seed", type=int, default=727000)
    p.add_argument("--extra-af3-seed", type=int, default=314159)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--oracle-exec", default=os.environ.get("ORACLE_EXEC", "/opt/af3env/bin/python"),
                   help="Python/wrapper executable for persistent run_alphafold_batch.py")
    p.add_argument("--project-root", type=Path, default=None,
                   help="Optional upstream RFD4 source root if configs require PROJECT_ROOT")
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    if args.stage == "_worker":
        return run_worker(args)

    work = args.work.resolve()
    done_file = work / "done.json"
    if done_file.is_file():
        print(f"Skipping completed monomer beam cohort {work}")
        return
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--stage", "_worker",
        "--length", str(args.length),
        "--replicates", str(args.replicates),
        "--gen-seed", str(args.gen_seed),
        "--internal-oracle-seed", str(args.internal_oracle_seed),
        "--assets", str(args.assets.resolve()),
        "--oracle-exec", str(args.oracle_exec),
        "--work", str(work),
    ]
    env = dict(
        os.environ,
        DATA_PATH=str(args.assets.resolve()),
        AF3_MODEL_DIR=str(args.assets.resolve() / "af3"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    if args.project_root is not None:
        env["PROJECT_ROOT"] = str(args.project_root.resolve())

    t0 = time.monotonic()
    with (work / "generation.log").open("w") as f:
        subprocess.run(cmd, env=env, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True)
    gen_wall_s = time.monotonic() - t0

    with (work / "outputs/inference_metrics/beam_example_per_example_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != args.replicates * 3:
        raise RuntimeError(f"Expected {args.replicates * 3} beam survivors (3 per replicate), got {len(rows)}")

    trace = json.loads((work / "beam_trace.json").read_text())
    terminal = [c for c in trace["checkpoints"] if c["step"] == 60]
    if len(terminal) != args.replicates:
        raise RuntimeError(f"Expected {args.replicates} step-60 checkpoint records, got {len(terminal)}")

    top1_paths = []
    for rep_idx, rnd in enumerate(terminal):
        best = max(range(len(rnd["reward"])), key=rnd["reward"].__getitem__)
        suffix = (
            f"monomer-{LENGTHS[args.length]}_rep{rep_idx}_bsc3_{best}"
            if args.replicates > 1 else f"monomer-{LENGTHS[args.length]}_bsc3_{best}"
        )
        matches = [r for r in rows if r["example_id"].endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Missing unique rank-0 survivor {suffix} in {[r['example_id'] for r in rows[:6]]}")
        cif_p = Path(matches[0]["designed_structure_path"])
        if not cif_p.is_file():
            raise RuntimeError(f"Missing survivor CIF {cif_p}")
        top1_paths.append(cif_p)

    batches = [json.loads(line) for line in (work / "sampling.jsonl").read_text().splitlines()]
    if len(batches) != args.replicates:
        raise RuntimeError(f"Expected {args.replicates} timed beam replicates, got {len(batches)}")
    warm_batches = batches[1:] if len(batches) > 1 else batches
    warm_s_per_rep = sum(b["seconds"] for b in warm_batches) / len(warm_batches)

    from score import score

    scored_primary = score({args.length: top1_paths}, args.replicates, args.assets.resolve(), work / "evaluation", args.af3_seed)
    scored_heldout = score(
        {args.length: top1_paths}, args.replicates, args.assets.resolve(),
        work / f"evaluation_seed_{args.extra_af3_seed}", args.extra_af3_seed,
    )

    summary = {
        "length": args.length,
        "replicates": args.replicates,
        "gen_seed": args.gen_seed,
        "internal_oracle_seed": args.internal_oracle_seed,
        "internal_af3_evaluations": args.replicates * 24,
        "generation_and_search_wall_s": gen_wall_s,
        "warm_generation_and_search_s_per_attempt": warm_s_per_rep,
        "batches": batches,
        "top1_paths": [str(q) for q in top1_paths],
        f"score_seed_{args.af3_seed}": scored_primary,
        f"score_seed_{args.extra_af3_seed}": scored_heldout,
    }
    done_file.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
