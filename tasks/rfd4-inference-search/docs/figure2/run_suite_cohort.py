"""Run an official Figure 2 target cohort (200 attempts) for any of the four suites (PPI, SM, Enzyme/MCSA, DNA)."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

SUITE_SPECS = {
    "ppi_backbone": {
        "track": "Protein binding",
        "criterion": "ppi",
        "metrics_tags": "[design,3d_coordinate_conditioning,hotspot_conditioning,ppi_refolding]",
        "oracle_key": "oracle_ppi",
        "schedules": {
            "Low": "low/front_loaded",
            "Medium": "medium/front_loaded",
            "High": "high/front_loaded",
        },
    },
    "studio_figure_2c_unfixed": {
        "track": "Small-molecule binding",
        "criterion": "sm_binder",
        "metrics_tags": "[design,sm_binder_refolding]",
        "oracle_key": "oracle",
        "schedules": {
            "Low": "low/midpeak",
            "Medium": "medium/midpeak",
            "High": "high/midpeak",
        },
    },
    "mcsa_fixed": {
        "track": "Enzyme scaffolding",
        "criterion": "ame",
        "metrics_tags": "[design,unindexed,ame_refolding]",
        "oracle_key": "oracle",
        "schedules": {
            "Low": "low/midpeak",
            "Medium": "medium/midpeak",
            "High": "high/midpeak",
        },
    },
    "dna_unfixed": {
        "track": "DNA binding",
        "criterion": "dna",
        "metrics_tags": "[design,dna_refolding,interchain_interactions]",
        "oracle_key": "oracle_dna",
        "schedules": {
            "Low": "low/front_loaded",
            "Medium": "medium/front_loaded",
            "High": "high/flat",
        },
    },
}


def input_stem(path: Path) -> str:
    name = path.name
    for suf in (".gz", ".zst", ".cif", ".pdb"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def run_gen_worker(args):
    import torch
    from rfproteina.generate import main_hydra
    from rfproteina.proteina import Proteina

    scratch = args.scratch.resolve()
    spec = SUITE_SPECS[args.dataset]
    orig_predict = Proteina.predict_step

    def timed_predict(self, batch, *a, **kw):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        res = orig_predict(self, batch, *a, **kw)
        torch.cuda.synchronize()
        with (scratch / "sampling.jsonl").open("a") as f:
            f.write(json.dumps({"size": int(batch["padding_mask"].shape[0]), "seconds": time.monotonic() - t0}) + "\n")
        return res

    Proteina.predict_step = timed_predict
    sys.argv = [
        "rfd4-generate",
        f"inputs={args.input_cif.resolve()}",
        f"out_dir={scratch / 'gen'}",
        f"hf_ckpt_path={args.assets.resolve() / 'rfd4'}",
        f"recycle_strategy={spec['schedules'][args.method]}",
        f"num_replicates={args.attempts}",
        f"batch_size={args.batch_size}",
        f"seed={args.gen_seed}",
        "compile_lengths=null",
        "num_workers=0",
        "skip_existing=false",
        "~metrics",
    ]
    main_hydra()


def run_eval_worker(args):
    import rfproteina
    import rfproteina.metrics.oracles.af3_oracle as af3_mod
    from rfproteina.evaluate import main_hydra

    scratch = args.scratch.resolve()
    assets = args.assets.resolve()
    spec = SUITE_SPECS[args.dataset]
    af3_timings = []
    warmup_compile_s = 0.0
    worker_proc = None
    worker_log = None
    local_warmed = False

    def wait_status():
        while True:
            line = worker_proc.stdout.readline()
            if not line:
                raise RuntimeError(f"AF3 worker exited with {worker_proc.poll()}")
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "status" in msg:
                return msg

    def dispatch_via_socket(manifest_path: str, n_designs: int):
        nonlocal warmup_compile_s
        req = json.dumps({"manifest_path": str(Path(manifest_path).resolve()), "n_designs": n_designs}) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(3600.0)
            s.connect(str(args.af3_sock))
            s.sendall(req.encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    raise RuntimeError("AF3 socket server closed connection unexpectedly")
                buf += chunk
        rep = json.loads(buf.decode("utf-8"))
        if rep.get("status") != "ok":
            raise RuntimeError(f"Persistent AF3 socket server failed: {rep}")
        warmup_compile_s += float(rep.get("warmup_s", 0.0))
        return float(rep["warm_s"])

    def dispatch(oracle, manifest_path, daemon_rank, n_designs):
        nonlocal worker_proc, worker_log, local_warmed, warmup_compile_s
        entries = json.loads(Path(manifest_path).read_text())
        if args.af3_sock is not None:
            warm_s = dispatch_via_socket(manifest_path, n_designs)
            af3_timings.append({"names": [e["name"] for e in entries], "n_designs": n_designs, "warm_af3_s": warm_s})
            return

        if worker_proc is None:
            t_start = time.monotonic()
            jax_cache = Path(os.environ.get("AF3_JAX_CACHE_DIR", str(scratch / "jax-cache")))
            jax_cache.mkdir(parents=True, exist_ok=True)
            worker_log = (scratch / "af3_worker.log").open("w")
            entry = Path(rfproteina.__file__).parent / "af3/run_alphafold_batch.py"
            cmd = [
                str(args.oracle_exec), str(entry), "--serve",
                "--num_recycles=5", "--num_diffusion_samples=1",
                "--buckets=256,320,384,448,512,576,640,768",
                "--flash_attention_implementation=xla", "--gpu_device=0",
                f"--model_dir={assets / 'af3'}",
                f"--jax_compilation_cache_dir={jax_cache}",
                "--save_full_outputs=false",
            ]
            worker_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True, bufsize=1,
            )
            if wait_status()["status"] != "ready":
                raise RuntimeError("AF3 worker failed startup")
            warmup_compile_s += time.monotonic() - t_start

        if not local_warmed:
            tc = time.monotonic()
            worker_proc.stdin.write(str(Path(manifest_path).resolve()) + "\n")
            worker_proc.stdin.flush()
            r0 = wait_status()
            if r0.get("status") != "ok" or r0.get("n_ok") != n_designs:
                raise RuntimeError(f"AF3 warmup fold failed: {r0}")
            warmup_compile_s += time.monotonic() - tc
            local_warmed = True

        t0 = time.monotonic()
        worker_proc.stdin.write(str(Path(manifest_path).resolve()) + "\n")
        worker_proc.stdin.flush()
        reply = wait_status()
        elapsed = time.monotonic() - t0
        if reply.get("status") != "ok" or reply.get("n_ok") != n_designs:
            raise RuntimeError(f"AF3 fold failed: {reply}")
        af3_timings.append({"names": [e["name"] for e in entries], "n_designs": n_designs, "warm_af3_s": elapsed})

    af3_mod.AF3Oracle._dispatch_fold = dispatch
    sys.argv = [
        "rfd4-evaluate",
        f"inputs={scratch / 'gen/inference_samples'}",
        f"task_csv={scratch / 'gen/inference_metrics/per_example_metrics.csv'}",
        f"out_dir={scratch / 'eval'}",
        f"metrics_tags={spec['metrics_tags']}",
        f"++metrics.refolding_oracles.{spec['oracle_key']}.seed={args.af3_seed}",
    ]
    try:
        main_hydra()
    finally:
        (scratch / "af3_timings.json").write_text(
            json.dumps({"warmup_compile_s": warmup_compile_s, "calls": af3_timings}, indent=2, allow_nan=False) + "\n"
        )
        if worker_proc is not None:
            try:
                worker_proc.stdin.write("shutdown\n")
                worker_proc.stdin.flush()
                worker_proc.wait(timeout=12)
            except Exception:
                worker_proc.kill()
                worker_proc.wait()
        if worker_log is not None:
            worker_log.close()


def cluster_chains(chain_paths: dict[str, Path], foldseek_bin: Path, folder: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    if not chain_paths:
        return {}
    if len(chain_paths) == 1:
        only = next(iter(chain_paths))
        return {only: only}
    struct_dir = folder / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    for key, p in chain_paths.items():
        (struct_dir / f"{key}.cif").symlink_to(p.resolve())
    with (folder / "foldseek.log").open("w") as log:
        subprocess.run(
            [
                str(foldseek_bin), "easy-cluster", str(struct_dir), str(folder / "clusters"),
                str(folder / "tmp"), "--alignment-type", "1", "--cov-mode", "0",
                "--min-seq-id", "0", "--tmscore-threshold", "0.5", "--threads", "4",
            ],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    with (folder / "clusters_cluster.tsv").open() as f:
        pairs = [(Path(rep).stem, Path(mem).stem) for rep, mem in csv.reader(f, delimiter="\t")]
    mapping = {m: r for r, m in pairs}
    if set(mapping) != set(chain_paths):
        raise RuntimeError("Foldseek cluster assignment mismatch on designed chains")
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "_gen", "_eval"), default="run")
    p.add_argument("--dataset", choices=tuple(SUITE_SPECS), required=True)
    p.add_argument("--method", choices=("Low", "Medium", "High"), required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--attempts", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--gen-seed", type=int, default=None)
    p.add_argument("--af3-seed", type=int, default=727000)
    p.add_argument("--input-cif", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--oracle-exec", default=os.environ.get("ORACLE_EXEC", "/opt/af3env/bin/python"))
    p.add_argument("--af3-sock", type=Path, default=None, help="Optional persistent AF3 Unix socket path")
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--scratch", type=Path, required=True, help="Fast ephemeral directory for per-target CIFs")
    p.add_argument("--out-dir", type=Path, required=True, help="Permanent output directory for target receipts/CSVs")
    p.add_argument("--mirror-dir", type=Path, default=None, help="Optional secondary mirror directory (e.g. on /usr0)")
    p.add_argument("--keep-scratch", action="store_true")
    args = p.parse_args()

    if args.stage == "_gen":
        if args.gen_seed is None:
            p.error("--gen-seed is required for _gen")
        return run_gen_worker(args)
    if args.stage == "_eval":
        return run_eval_worker(args)
    if args.gen_seed is None:
        p.error("--gen-seed is required for run")

    drain_file = os.environ.get("RFD4_DRAIN_FILE", "/dev/shm/rfd4-fig2-scratch/DRAIN_FOR_HANDOVER")
    while drain_file and Path(drain_file).exists():
        time.sleep(2.0)

    out_dir = args.out_dir.resolve()
    done_file = out_dir / "done.json"
    if done_file.is_file():
        print(f"Skipping completed cohort {out_dir}", flush=True)
        return

    scratch = args.scratch.resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(
        os.environ,
        DATA_PATH=str(args.assets.resolve()),
        AF3_MODEL_DIR=str(args.assets.resolve() / "af3"),
        FOLDSEEK_EXEC=str(args.assets.resolve() / "foldseek"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TMPDIR=str(scratch),
    )
    if args.project_root is not None:
        env["PROJECT_ROOT"] = str(args.project_root.resolve())

    try:
        # 1. Generation
        gen_cmd = [
            sys.executable, str(Path(__file__).resolve()), "--stage", "_gen",
            "--dataset", args.dataset, "--method", args.method, "--target", args.target,
            "--attempts", str(args.attempts), "--batch-size", str(args.batch_size),
            "--gen-seed", str(args.gen_seed), "--input-cif", str(args.input_cif.resolve()),
            "--assets", str(args.assets.resolve()), "--scratch", str(scratch), "--out-dir", str(out_dir),
        ]
        t0 = time.monotonic()
        with (out_dir / "gen.log").open("w") as f:
            subprocess.run(gen_cmd, env=env, cwd=scratch, stdout=f, stderr=subprocess.STDOUT, check=True)
        gen_wall_s = time.monotonic() - t0

        batches = [json.loads(line) for line in (scratch / "sampling.jsonl").read_text().splitlines()]
        if sum(b["size"] for b in batches) != args.attempts:
            raise RuntimeError(f"Expected {args.attempts} generated attempts, got {sum(b['size'] for b in batches)}")

        warm_batches = batches[1:] if len(batches) > 1 else batches
        mean_warm_gen_s = sum(b["seconds"] for b in warm_batches) / sum(b["size"] for b in warm_batches)
        per_attempt_warm_gen_s = []
        for idx_b, b in enumerate(batches):
            per_item = mean_warm_gen_s if (idx_b == 0 and len(batches) > 1) else (b["seconds"] / b["size"])
            per_attempt_warm_gen_s.extend([per_item] * b["size"])

        # 2. Evaluation (common AF3)
        eval_cmd = [
            sys.executable, str(Path(__file__).resolve()), "--stage", "_eval",
            "--dataset", args.dataset, "--method", args.method, "--target", args.target,
            "--af3-seed", str(args.af3_seed), "--input-cif", str(args.input_cif.resolve()),
            "--assets", str(args.assets.resolve()), "--oracle-exec", str(args.oracle_exec),
            "--scratch", str(scratch), "--out-dir", str(out_dir),
        ]
        if args.af3_sock is not None:
            eval_cmd.extend(["--af3-sock", str(args.af3_sock.resolve())])
        t1 = time.monotonic()
        with (out_dir / "eval.log").open("w") as f:
            subprocess.run(eval_cmd, env=env, cwd=scratch, stdout=f, stderr=subprocess.STDOUT, check=True)
        eval_wall_s = time.monotonic() - t1

        eval_csv_path = scratch / "eval/evaluation_metrics/per_example_metrics.csv"
        with (scratch / "gen/inference_metrics/per_example_metrics.csv").open() as f:
            gen_rows = list(csv.DictReader(f))
        with eval_csv_path.open() as f:
            eval_rows = list(csv.DictReader(f))
        if len(gen_rows) != args.attempts or len(eval_rows) != args.attempts:
            raise RuntimeError(f"Row count mismatch: gen={len(gen_rows)}, eval={len(eval_rows)}")

        # Preserve full compressed evaluation metrics CSV (~8 KB) permanently
        with eval_csv_path.open("rb") as fin, gzip.open(out_dir / "per_example_metrics.csv.gz", "wb") as fout:
            shutil.copyfileobj(fin, fout)

        eval_by_path = {str(Path(r["designed_structure_path"]).resolve()): r for r in eval_rows}
        af3_timing_doc = json.loads((scratch / "af3_timings.json").read_text())
        af3_time_by_name = {}
        for c in af3_timing_doc["calls"]:
            per_job = c["warm_af3_s"] / c["n_designs"]
            for nm in c["names"]:
                af3_time_by_name[nm] = per_job

        # 3. Foldseek clustering on designed_chain_path (§S12)
        stem = input_stem(args.input_cif)
        chain_map = {}
        ordered_eval = []
        gen_by_eid = {g["example_id"]: g for g in gen_rows}
        for i in range(args.attempts):
            rep_id = f"{stem}_rep{i}" if args.attempts > 1 else stem
            if rep_id not in gen_by_eid:
                raise RuntimeError(f"Missing generated row {rep_id} in {list(gen_by_eid)[:4]}")
            g_row = gen_by_eid[rep_id]
            e_row = eval_by_path[str(Path(g_row["designed_structure_path"]).resolve())]
            ordered_eval.append((i, rep_id, e_row))
            cpath = e_row.get("designed_chain_path", "")
            if cpath and Path(cpath).is_file():
                chain_map[f"a{i:03d}"] = Path(cpath)

        clusters = cluster_chains(chain_map, args.assets.resolve() / "foldseek", scratch / "foldseek")
        with (out_dir / "clusters.tsv").open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["attempt_slot", "cluster_rep"])
            for k in sorted(clusters):
                w.writerow([k, clusters[k]])

        # 4. Build per-attempt warm & cold trial ledgers
        crit = SUITE_SPECS[args.dataset]["criterion"]
        warm_trials, cold_trials = [], []
        cold_gen_each = gen_wall_s / args.attempts
        scored_count = sum(1 for _, _, er in ordered_eval if float(er["pre_oracle_pass"]) == 1.0)
        cold_af3_each = (eval_wall_s / scored_count) if scored_count > 0 else 0.0

        # For MCSA, also audit §S8.2 no-clash variant alongside upstream ame.pass
        ame_noclash_passes = 0
        ame_noclash_clusters = set()

        oracle_key = SUITE_SPECS[args.dataset]["oracle_key"]
        for i, _, er in ordered_eval:
            pre_ok = int(float(er["pre_oracle_pass"]))
            complete = int(float(er[f"{crit}.all_metrics_present"]))
            passed = int(float(er[f"{crit}.pass"]))
            if pre_ok and not complete:
                # Distinguish a genuine AF3 runtime failure (missing overall_plddt) from an unmatched
                # post-refold motif/geometry calculation (Supplement §S8.2: unmatched required motifs fail).
                plddt_str = str(er.get(f"{oracle_key}.overall_plddt", "")).strip()
                if plddt_str and math.isfinite(float(plddt_str)):
                    complete = 1
                    passed = 0
                else:
                    raise RuntimeError(f"Indeterminate AF3 metrics for {er['example_id']}")
            scored = 1 if (pre_ok and complete) else 0
            eid = er["example_id"]
            warm_af3_s = af3_time_by_name[eid] if pre_ok else 0.0
            cid = clusters.get(f"a{i:03d}", "") if passed else ""

            if args.dataset == "mcsa_fixed" and scored:
                try:
                    ipae = float(er["oracle.min_ipae_coi"])
                    bb_r = float(er["refolded_backbone_rmsd.rmsd"])
                    mot_r = float(er["refolded_motif_allatom_rmsd.rmsd"])
                    if ipae < 1.5 and bb_r <= 2.0 and mot_r <= 1.5:
                        ame_noclash_passes += 1
                        if f"a{i:03d}" in clusters:
                            ame_noclash_clusters.add(clusters[f"a{i:03d}"])
                except (KeyError, ValueError):
                    pass

            base_row = {
                "dataset": args.dataset,
                "criterion": crit,
                "target": args.target,
                "attempt_id": f"rep{i:03d}",
                "scored": scored,
                "passed": passed,
                "cluster_id": cid,
                "pre_oracle_pass": pre_ok,
                "all_metrics_present": complete,
            }
            warm_trials.append({
                **base_row,
                "method": args.method,
                "generation_gpu_s": per_attempt_warm_gen_s[i],
                "af3_gpu_s": warm_af3_s,
            })
            cold_trials.append({
                **base_row,
                "method": f"{args.method} (cold)",
                "generation_gpu_s": cold_gen_each,
                "af3_gpu_s": cold_af3_each if pre_ok else 0.0,
            })

        fields = (
            "dataset", "criterion", "method", "target", "attempt_id", "scored", "passed",
            "cluster_id", "generation_gpu_s", "af3_gpu_s", "pre_oracle_pass", "all_metrics_present",
        )
        for fname, rows in (("trials_warm.csv", warm_trials), ("trials_cold.csv", cold_trials)):
            with (out_dir / fname).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)

        passes = sum(r["passed"] for r in warm_trials)
        unique_clusters = len({r["cluster_id"] for r in warm_trials if r["passed"]})
        warm_gen_sum = math.fsum(r["generation_gpu_s"] for r in warm_trials)
        warm_af3_sum = math.fsum(r["af3_gpu_s"] for r in warm_trials)

        summary = {
            "profile": "rfd4-a6000-warm-v1",
            "dataset": args.dataset,
            "criterion": crit,
            "method": args.method,
            "target": args.target,
            "attempts": args.attempts,
            "batch_size": args.batch_size,
            "gen_seed": args.gen_seed,
            "af3_seed": args.af3_seed,
            "scored": scored_count,
            "passes": passes,
            "clusters": unique_clusters,
            "clustered_pass_pct": 100.0 * unique_clusters / args.attempts,
            "generation_gpu_s": warm_gen_sum,
            "af3_gpu_s": warm_af3_sum,
            "cold_generation_wall_s": gen_wall_s,
            "cold_eval_wall_s": eval_wall_s,
            "af3_warmup_compile_s": af3_timing_doc["warmup_compile_s"],
            "batches": batches,
        }
        if args.dataset == "mcsa_fixed":
            summary["ame_s8_no_clash_filter_passes"] = ame_noclash_passes
            summary["ame_s8_no_clash_filter_clusters"] = len(ame_noclash_clusters)

        tmp_done = out_dir / "done.json.tmp"
        tmp_done.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        tmp_done.replace(done_file)

        if args.mirror_dir is not None:
            mdir = args.mirror_dir.resolve()
            mdir.mkdir(parents=True, exist_ok=True)
            for fn in ("done.json", "trials_warm.csv", "trials_cold.csv", "clusters.tsv", "per_example_metrics.csv.gz"):
                shutil.copy2(out_dir / fn, mdir / fn)
    finally:
        if not args.keep_scratch and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
