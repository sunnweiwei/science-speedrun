"""End-to-end 4-track paired evaluator and strict Pareto frontier grader for unified RFD4 strategies.

Executes a single submission package (`infer.py --request FILE --output DIRECTORY`) alongside paired
executable human baselines (`native-low`, `native-medium`, `native-high`) across all four Figure 2 tracks
(Protein binding, Small-molecule binding, Enzyme scaffolding, DNA binding) on the same GPU, same target
roster, same attempt count, same seeds, and identical subprocess + persistent-AF3 timing boundaries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import socket
from statistics import fmean, stdev
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

OVERALL_CANDIDATES = (
    Path("/opt/rfd4-overall"),
    HERE.parent / "references/overall",
)
OVERALL = next((p for p in OVERALL_CANDIDATES if (p / "benchmark_curve.py").is_file()), OVERALL_CANDIDATES[-1])
if str(OVERALL) not in sys.path:
    sys.path.insert(0, str(OVERALL))

from benchmark_curve import ARMS, Measurement, TRACKS, frontier, read_targets, summarize
from policy import InvalidPolicy
from program import gpu_device, package_digest, run_conditioned_program, verifier_runtime, visible_gpus
from reference import NATIVE_PRESETS, package_reference
from requests_io import DEV_4TRACK_TARGETS, conditioned_suite_requests

PAIRED_TIMING_PROFILE = "rfd4-paired-program-v1"
WARM_CAMPAIGN_PROFILE = "rfd4-a6000-warm-v1"
AUTHOR_ARCHIVE_PROFILE = "author-h200-archive-v1"

PAIRED_REFERENCE_LABELS = {
    "native-low": "Low",
    "native-medium": "Medium",
    "native-high": "High",
}

TRACK_SPECS = {
    "ppi_backbone": {
        "track": "Protein binding",
        "short_key": "ppi",
        "criterion": "ppi",
        "metrics_tags": "[design,3d_coordinate_conditioning,hotspot_conditioning,ppi_refolding]",
        "oracle_key": "oracle_ppi",
        "seed_base": 727201,
    },
    "studio_figure_2c_unfixed": {
        "track": "Small-molecule binding",
        "short_key": "sm_binder",
        "criterion": "sm_binder",
        "metrics_tags": "[design,sm_binder_refolding]",
        "oracle_key": "oracle",
        "seed_base": 727301,
    },
    "mcsa_fixed": {
        "track": "Enzyme scaffolding",
        "short_key": "ame",
        "criterion": "ame",
        "metrics_tags": "[design,unindexed,ame_refolding]",
        "oracle_key": "oracle",
        "seed_base": 727101,
    },
    "dna_unfixed": {
        "track": "DNA binding",
        "short_key": "dna",
        "criterion": "dna",
        "metrics_tags": "[design,dna_refolding,interchain_interactions]",
        "oracle_key": "oracle_dna",
        "seed_base": 727001,
    },
}
TRACK_TO_SHORT_KEY = {spec["track"]: spec["short_key"] for spec in TRACK_SPECS.values()}


def _af3_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    af3_src = Path("/opt/af3/src")
    if af3_src.is_dir():
        cur = env.get("PYTHONPATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if str(af3_src) not in parts:
            env["PYTHONPATH"] = os.pathsep.join([str(af3_src), *parts])
    return env


def _estimate_af3_bucket(manifest_path: Path) -> int:
    entries = json.loads(manifest_path.read_text())
    max_tok = 0
    for e in entries:
        toks = 0
        for seq_obj in e.get("sequences", []):
            if "protein" in seq_obj:
                toks += len(seq_obj["protein"].get("sequence", ""))
            elif "dna" in seq_obj:
                toks += len(seq_obj["dna"].get("sequence", ""))
            elif "rna" in seq_obj:
                toks += len(seq_obj["rna"].get("sequence", ""))
            elif "ligand" in seq_obj:
                lig = seq_obj["ligand"]
                if "smiles" in lig:
                    toks += max(8, len(lig["smiles"]) // 2)
                elif "ccdCodes" in lig:
                    toks += 16 * len(lig["ccdCodes"])
                else:
                    toks += 24
        max_tok = max(max_tok, toks)
    for b in (256, 320, 384, 448, 512, 576, 640, 768, 1024):
        if max_tok <= b:
            return b
    return 1024


def _run_af3_server(args):
    """Persistent AF3 batch socket server shared across candidate and paired references."""
    import rfproteina

    args.jax_cache.mkdir(parents=True, exist_ok=True)
    args.af3_log.parent.mkdir(parents=True, exist_ok=True)
    if args.af3_sock.exists():
        args.af3_sock.unlink()
    args.af3_sock.parent.mkdir(parents=True, exist_ok=True)

    entry = Path(rfproteina.__file__).parent / "af3/run_alphafold_batch.py"
    cmd = [
        str(Path(args.oracle_exec).resolve()), str(entry), "--serve",
        "--num_recycles=5", "--num_diffusion_samples=1",
        "--buckets=256,320,384,448,512,576,640,768",
        "--flash_attention_implementation=xla", "--gpu_device=0",
        f"--model_dir={(args.assets.resolve() / 'af3')}",
        f"--jax_compilation_cache_dir={args.jax_cache.resolve()}",
        "--save_full_outputs=false",
    ]
    wlog = args.af3_log.open("a")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=wlog,
        env=_af3_subprocess_env(), text=True, bufsize=1,
    )

    def wait_msg():
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"AF3 batch worker exited with {proc.poll()}")
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "status" in msg:
                return msg

    if wait_msg().get("status") != "ready":
        raise RuntimeError("Persistent AF3 batch worker failed to reach ready state")

    warmed_buckets: set[int] = set()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(args.af3_sock))
    srv.listen(8)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf:
                    continue
                req = json.loads(buf.decode("utf-8"))
                if req.get("cmd") == "ping":
                    conn.sendall(b'{"status":"ready"}\n')
                    continue
                if req.get("cmd") == "shutdown":
                    conn.sendall(b'{"status":"bye"}\n')
                    break
                mpath = Path(req["manifest_path"])
                n_designs = int(req["n_designs"])
                bucket = _estimate_af3_bucket(mpath)
                warmup_s = 0.0
                if bucket not in warmed_buckets:
                    tw = time.monotonic()
                    proc.stdin.write(str(mpath) + "\n")
                    proc.stdin.flush()
                    r0 = wait_msg()
                    if r0.get("status") != "ok" or r0.get("n_ok") != n_designs:
                        conn.sendall((json.dumps({"status": "error", "detail": r0}) + "\n").encode())
                        continue
                    warmup_s = time.monotonic() - tw
                    warmed_buckets.add(bucket)
                t0 = time.monotonic()
                proc.stdin.write(str(mpath) + "\n")
                proc.stdin.flush()
                rep = wait_msg()
                warm_s = time.monotonic() - t0
                if rep.get("status") != "ok" or rep.get("n_ok") != n_designs:
                    conn.sendall((json.dumps({"status": "error", "detail": rep}) + "\n").encode())
                else:
                    conn.sendall((json.dumps({
                        "status": "ok",
                        "n_ok": n_designs,
                        "bucket": bucket,
                        "warmup_s": warmup_s,
                        "warm_s": warm_s,
                    }) + "\n").encode())
    finally:
        srv.close()
        if args.af3_sock.exists():
            args.af3_sock.unlink()
        try:
            proc.stdin.write("shutdown\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        wlog.close()


@contextmanager
def managed_af3_server(assets: Path, oracle_exec: str, work: Path, external_sock: Path | None = None, gpu_info: dict | None = None):
    """Ensure candidate and all paired baselines share one warm AF3 server with bucket warmup excluded."""
    if external_sock is not None:
        yield external_sock.resolve()
        return
    sock_dir = Path(tempfile.mkdtemp(prefix="rfd4-af3s-"))
    sock_path = sock_dir / "af3.sock"
    gpu_tag = f"gpu{gpu_info['index']}_" if gpu_info is not None and "index" in gpu_info else ""
    srv_log = work / f"af3_shared_server_{gpu_tag}log.txt"
    jax_cache = Path(os.environ.get("AF3_JAX_CACHE_DIR", str(work / "jax-cache")))
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--stage", "_af3_server",
        "--assets", str(assets.resolve()), "--oracle-exec", str(oracle_exec),
        "--jax-cache", str(jax_cache.resolve()), "--af3-sock", str(sock_path),
        "--af3-log", str(srv_log.resolve()),
    ]
    srv_env = dict(os.environ)
    if gpu_info is not None:
        srv_env["CUDA_VISIBLE_DEVICES"] = gpu_info["uuid"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=srv_env)
    try:
        deadline = time.monotonic() + 300.0
        while True:
            if proc.poll() is not None:
                detail = srv_log.read_text()[-2000:] if srv_log.is_file() else ""
                raise RuntimeError(f"Shared AF3 server exited prematurely ({proc.returncode}): {detail}")
            if sock_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.settimeout(5.0)
                        s.connect(str(sock_path))
                        s.sendall(b'{"cmd":"ping"}\n')
                        if json.loads(s.recv(4096).decode("utf-8")).get("status") == "ready":
                            break
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for shared AF3 batch server")
            time.sleep(0.25)
        yield sock_path
    finally:
        if proc.poll() is None and sock_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(10.0)
                    s.connect(str(sock_path))
                    s.sendall(b'{"cmd":"shutdown"}\n')
                    s.recv(1024)
            except OSError:
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(sock_dir, ignore_errors=True)


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


def _run_eval_worker(args):
    import rfproteina
    import rfproteina.metrics.oracles.af3_oracle as af3_mod
    from rfproteina.evaluate import main_hydra

    scratch = args.scratch.resolve()
    assets = args.assets.resolve()
    spec = TRACK_SPECS[args.dataset]
    af3_timings = []
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

    def dispatch(oracle, manifest_path, daemon_rank, n_designs):
        nonlocal worker_proc, worker_log, local_warmed
        entries = json.loads(Path(manifest_path).read_text())
        if args.af3_sock is not None:
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
            af3_timings.append({"names": [e["name"] for e in entries], "n_designs": n_designs, "warm_af3_s": float(rep["warm_s"])})
            return

        if worker_proc is None:
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
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log,
                env=_af3_subprocess_env(), text=True, bufsize=1,
            )
            if wait_status()["status"] != "ready":
                raise RuntimeError("AF3 worker failed startup")

        if not local_warmed:
            worker_proc.stdin.write(str(Path(manifest_path).resolve()) + "\n")
            worker_proc.stdin.flush()
            r0 = wait_status()
            if r0.get("status") != "ok" or r0.get("n_ok") != n_designs:
                raise RuntimeError(f"AF3 warmup fold failed: {r0}")
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
        f"inputs={scratch / 'staged_samples'}",
        f"task_csv={scratch / 'task_map.csv'}",
        f"out_dir={scratch / 'eval'}",
        f"metrics_tags={spec['metrics_tags']}",
        f"++metrics.refolding_oracles.{spec['oracle_key']}.seed={args.af3_seed}",
    ]
    try:
        main_hydra()
    finally:
        (scratch / "af3_timings.json").write_text(json.dumps({"calls": af3_timings}, indent=2, allow_nan=False) + "\n")
        if worker_proc is not None:
            try:
                worker_proc.stdin.write("shutdown\n")
                worker_proc.stdin.flush()
                worker_proc.wait(timeout=10)
            except Exception:
                worker_proc.kill()
        if worker_log is not None:
            worker_log.close()


def score_target_slots(
    dataset: str,
    target: str,
    method_name: str,
    slot_paths: list[Path | None],
    gen_seconds: float,
    assets: Path,
    work_dir: Path,
    af3_seed: int,
    oracle_exec: str,
    af3_sock: Path | None = None,
    project_root: Path | None = None,
    repeat_index: int = 0,
    gpu_info: dict | None = None,
) -> tuple[Measurement, list[dict]]:
    """Score emitted slot-*.cif structures on one official Figure 2 target."""
    spec = TRACK_SPECS[dataset]
    crit = spec["criterion"]
    oracle_key = spec["oracle_key"]
    attempts = len(slot_paths)
    per_attempt_gen_s = gen_seconds / attempts

    staged = work_dir / "staged_samples"
    staged.mkdir(parents=True, exist_ok=True)
    valid_slots: dict[int, Path] = {}
    task_rows = []
    for idx, sp in enumerate(slot_paths):
        if sp is None or not sp.is_file() or sp.stat().st_size == 0:
            continue
        dst = staged / f"{target}_slot{idx:04d}.cif"
        shutil.copyfile(sp, dst)
        valid_slots[idx] = dst.resolve()
        task_rows.append({"designed_structure_path": str(dst.resolve()), "task": target})

    if not valid_slots:
        trials = [{
            "repeat": repeat_index, "dataset": dataset, "criterion": crit, "method": method_name,
            "target": target, "attempt_id": f"slot{i:04d}", "scored": 0, "passed": 0, "cluster_id": "",
            "generation_gpu_s": per_attempt_gen_s, "af3_gpu_s": 0.0,
        } for i in range(attempts)]
        return Measurement(attempts, 0, 0, 0, gen_seconds, 0.0), trials

    with (work_dir / "task_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["designed_structure_path", "task"], lineterminator="\n")
        w.writeheader()
        w.writerows(task_rows)

    env = dict(
        os.environ,
        DATA_PATH=str(assets.resolve()),
        AF3_MODEL_DIR=str(assets.resolve() / "af3"),
        FOLDSEEK_EXEC=str(assets.resolve() / "foldseek"),
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TMPDIR=str(work_dir.resolve()),
    )
    if project_root is not None:
        env["PROJECT_ROOT"] = str(project_root.resolve())
    if gpu_info is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_info["uuid"]

    eval_cmd = [
        sys.executable, str(Path(__file__).resolve()), "--stage", "_eval",
        "--dataset", dataset, "--af3-seed", str(af3_seed),
        "--assets", str(assets.resolve()), "--oracle-exec", str(oracle_exec),
        "--scratch", str(work_dir.resolve()),
    ]
    if af3_sock is not None:
        eval_cmd.extend(["--af3-sock", str(af3_sock.resolve())])
    with (work_dir / "eval.log").open("w") as f:
        subprocess.run(eval_cmd, env=env, cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, check=True)

    with (work_dir / "eval/evaluation_metrics/per_example_metrics.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    eval_by_path = {str(Path(r["designed_structure_path"]).resolve()): r for r in eval_rows}

    af3_doc = json.loads((work_dir / "af3_timings.json").read_text())
    af3_time_by_eid = {}
    for c in af3_doc["calls"]:
        each_s = c["warm_af3_s"] / c["n_designs"]
        for nm in c["names"]:
            af3_time_by_eid[nm] = each_s

    chain_map = {}
    for idx, staged_path in valid_slots.items():
        er = eval_by_path.get(str(staged_path))
        if er is None:
            continue
        cpath = er.get("designed_chain_path", "")
        if cpath and Path(cpath).is_file():
            chain_map[f"s{idx:04d}"] = Path(cpath)

    clusters = cluster_chains(chain_map, assets.resolve() / "foldseek", work_dir / "foldseek")
    trials = []
    scored_total = passes_total = 0
    af3_sum = 0.0
    passing_clusters = set()

    for idx in range(attempts):
        staged_path = valid_slots.get(idx)
        er = eval_by_path.get(str(staged_path)) if staged_path else None
        if er is None:
            trials.append({
                "repeat": repeat_index, "dataset": dataset, "criterion": crit, "method": method_name,
                "target": target, "attempt_id": f"slot{idx:04d}", "scored": 0, "passed": 0, "cluster_id": "",
                "generation_gpu_s": per_attempt_gen_s, "af3_gpu_s": 0.0,
            })
            continue
        pre_ok = int(float(er["pre_oracle_pass"]))
        complete = int(float(er[f"{crit}.all_metrics_present"]))
        passed = int(float(er[f"{crit}.pass"]))
        if pre_ok and not complete:
            plddt_str = str(er.get(f"{oracle_key}.overall_plddt", "")).strip()
            if plddt_str and math.isfinite(float(plddt_str)):
                complete = 1
                passed = 0
            else:
                raise RuntimeError(f"Indeterminate AF3 metrics for {er['example_id']}")
        scored = 1 if (pre_ok and complete) else 0
        af3_s = af3_time_by_eid[er["example_id"]] if pre_ok else 0.0
        cid = clusters.get(f"s{idx:04d}", "") if passed else ""
        if passed and cid:
            passing_clusters.add(cid)
        scored_total += scored
        passes_total += passed
        af3_sum += af3_s
        trials.append({
            "repeat": repeat_index, "dataset": dataset, "criterion": crit, "method": method_name,
            "target": target, "attempt_id": f"slot{idx:04d}", "scored": scored, "passed": passed, "cluster_id": cid,
            "generation_gpu_s": per_attempt_gen_s, "af3_gpu_s": af3_s,
        })

    m = Measurement(attempts, scored_total, passes_total, len(passing_clusters), gen_seconds, af3_sum)
    return m, trials


def load_archive_context_for_targets(
    selected_targets: dict[str, list[str]],
    reproduced_csv: Path | None = None,
    author_csv: Path | None = None,
) -> dict[str, list[dict]]:
    """Return 200-attempt archive coordinates strictly as background context (never mixed with K != 200 or subprocess runs)."""
    reproduced_csv = reproduced_csv or (OVERALL / "reproduced_targets.csv")
    author_csv = author_csv or (OVERALL / "targets.csv")
    local_groups = read_targets(reproduced_csv) if reproduced_csv.is_file() else {}
    author_groups = read_targets(author_csv)

    by_track: dict[str, list[dict]] = {}
    for ds, tgts in selected_targets.items():
        track = TRACK_SPECS[ds]["track"]
        tset = set(tgts)
        pts = []
        for method in ARMS:
            loc = local_groups.get((track, method), {})
            if tset.issubset(loc.keys()):
                pt = summarize(track, method, {t: loc[t] for t in tgts}, "local warm archive (200 att/target)")
                pt["timing_profile"] = WARM_CAMPAIGN_PROFILE
                pt["attempts_per_target"] = 200
                pts.append(pt)
            elif (track, method) in author_groups and tset.issubset(author_groups[track, method].keys()):
                pt = summarize(track, method, {t: author_groups[track, method][t] for t in tgts}, "author archive (200 att/target)")
                pt["timing_profile"] = AUTHOR_ARCHIVE_PROFILE
                pt["attempts_per_target"] = 200
                pts.append(pt)
        by_track[track] = pts
    return by_track


def summarize_archive_200_quality(
    summary: dict,
    archive_by_track: dict[str, list[dict]],
    *,
    attempts_per_target: int,
) -> dict[str, dict]:
    """Compare K=200 candidate clustered pass rates against target-matched 200-attempt archive quality (quality only, never timing)."""
    if int(attempts_per_target) != 200:
        return {}
    eval_tracks = summary.get("evaluated_tracks") or []
    if not eval_tracks:
        return {}
    cand_by_track = {t: summary["by_track"][t]["candidate"] for t in eval_tracks if t in summary.get("by_track", {})}
    if len(cand_by_track) != len(eval_tracks):
        return {}
    out: dict[str, dict] = {}
    for method in ARMS:
        arch_pts = {}
        for t in eval_tracks:
            matched = next((p for p in archive_by_track.get(t, []) if p.get("method") == method), None)
            if matched is not None:
                arch_pts[t] = matched
        if len(arch_pts) != len(eval_tracks):
            continue
        arch_macro = fmean(float(arch_pts[t]["clustered_pct"]) for t in eval_tracks)
        cand_macro = float(summary["macro_clustered_pct"])
        out[method] = {
            "archive_macro_clustered_pct": arch_macro,
            "quality_delta_pp": cand_macro - arch_macro,
            "tracks_beating_archive": sum(
                int(float(cand_by_track[t]["clustered_pct"]) > float(arch_pts[t]["clustered_pct"]))
                for t in eval_tracks
            ),
            "by_track_clustered_pct": {t: float(arch_pts[t]["clustered_pct"]) for t in eval_tracks},
        }
    return out


def hypervolume_2d(points: list[dict], speed_axis: str, ref_speed: float = 0.0, ref_quality: float = 0.0) -> float:
    """Exact 2D dominated hypervolume above (ref_speed, ref_quality) on a maximize-(speed, clustered_pct) plane."""
    valid = [
        p for p in points
        if float(p[speed_axis]) > ref_speed and float(p["clustered_pct"]) > ref_quality
    ]
    if not valid:
        return 0.0
    nd = frontier(valid, speed_axis)
    # Deduplicate identical or tied speed coordinates by keeping max quality at each distinct speed.
    by_speed: dict[float, float] = {}
    for p in nd:
        x = float(p[speed_axis])
        y = float(p["clustered_pct"])
        by_speed[x] = max(by_speed.get(x, 0.0), y)
    ordered = sorted(by_speed.items(), key=lambda kv: kv[0])
    area = 0.0
    prev_x = ref_speed
    for x, y in ordered:
        area += (x - prev_x) * (y - ref_quality)
        prev_x = x
    return area


def summarize_paired_baselines(
    paired_measurements_by_ref: dict[str, dict[str, dict[str, Measurement]]],
    timing_profile: str = PAIRED_TIMING_PROFILE,
    ref_complete_by_track: dict[tuple[str, str], bool] | None = None,
) -> dict[str, list[dict]]:
    """Summarize live paired reference baselines under the exact same target roster, attempt count, and timing boundary."""
    by_track: dict[str, list[dict]] = {}
    for ref_key, track_map in paired_measurements_by_ref.items():
        label = PAIRED_REFERENCE_LABELS.get(ref_key, ref_key)
        for track, tgt_map in track_map.items():
            pt = summarize(track, label, tgt_map, f"paired {ref_key}")
            pt["reference_package"] = ref_key
            pt["timing_profile"] = timing_profile
            pt["complete"] = (
                True if ref_complete_by_track is None else bool(ref_complete_by_track.get((ref_key, track), True))
            )
            by_track.setdefault(track, []).append(pt)
    return by_track


def pareto_summary_across_tracks(
    candidate_by_track: dict[str, dict[str, Measurement]],
    baseline_points_by_track: dict[str, list[dict]],
    method_name: str = "submission",
    *,
    timing_profile: str = PAIRED_TIMING_PROFILE,
    candidate_complete_by_track: dict[str, bool] | None = None,
    references_complete_by_track: dict[str, bool] | None = None,
) -> dict:
    """Evaluate candidate strategy quality, matched-boundary throughput, and strict Pareto advancement.

    Enforces matched target counts, matched attempt counts, and matched timing boundaries between the
    candidate and every baseline point used for Pareto or dominance claims. Distinguishes weak
    nondominance (`on_*_pareto`, True on exact ties) from strict Pareto advancement
    (`strictly_advances_*_pareto` and `*_hypervolume_gain_pct`, strictly False/0.0 on exact ties).
    """
    track_reports = {}
    all_rows = []
    cand_gen_s = 0.0
    cand_af3_s = 0.0

    for track, cand_targets in candidate_by_track.items():
        cand_gen_s += math.fsum(m.generation for m in cand_targets.values())
        cand_af3_s += math.fsum(m.af3 for m in cand_targets.values())
        cand_pt = summarize(track, method_name, cand_targets, "candidate strategy")
        cand_pt["timing_profile"] = timing_profile
        cand_complete = True if candidate_complete_by_track is None else bool(candidate_complete_by_track.get(track, True))
        refs_complete = True if references_complete_by_track is None else bool(references_complete_by_track.get(track, True))
        cand_pt["complete"] = cand_complete

        base_pts = [dict(b) for b in baseline_points_by_track.get(track, [])]
        for b in base_pts:
            if int(b["targets"]) != int(cand_pt["targets"]) or int(b["attempts"]) != int(cand_pt["attempts"]):
                raise ValueError(
                    f"{track}: cannot compare candidate ({cand_pt['targets']} targets, {cand_pt['attempts']} attempts) "
                    f"against baseline {b['method']!r} ({b['targets']} targets, {b['attempts']} attempts); "
                    "clustered pass rate depends on sample count"
                )
            b_prof = b.get("timing_profile")
            if b_prof is not None and b_prof != timing_profile:
                raise ValueError(
                    f"{track}: cannot compare candidate timing profile {timing_profile!r} "
                    f"against baseline {b['method']!r} with timing profile {b_prof!r}"
                )
            b.setdefault("timing_profile", timing_profile)
            if "complete" not in b:
                b["complete"] = refs_complete

        all_refs_complete = refs_complete and all(bool(b.get("complete", True)) for b in base_pts)
        comparable = bool(base_pts) and cand_complete and all_refs_complete
        quality_floor = min((b["clustered_pct"] for b in base_pts), default=0.0)

        for kind, axis in (("generation", "generation_attempts_h"), ("total", "total_attempts_h")):
            if not comparable:
                cand_pt[f"on_{kind}_pareto"] = False
                cand_pt[f"strictly_advances_{kind}_pareto"] = False
                cand_pt[f"dominates_any_reference_{kind}"] = False
                cand_pt[f"dominates_a_baseline_on_{axis}"] = False
                cand_pt[f"dominates_strongest_reference_{kind}"] = False
                cand_pt[f"{kind}_hypervolume_gain_pct"] = 0.0
                cand_pt[f"gain_pp_at_matched_or_faster_{axis}"] = None
                continue

            combined = base_pts + [cand_pt]
            nd = {id(p) for p in frontier(combined, axis)}
            # Weak nondominance: True for exact baseline duplicates; never interpret alone as advancing the frontier.
            cand_pt[f"on_{kind}_pareto"] = id(cand_pt) in nd

            # Strict Pareto frontier advancement: candidate must meet at least the lowest baseline quality floor
            # and strictly beat every paired baseline on either speed or clustered pass rate (False for clones).
            strictly_advances = (
                cand_pt["clustered_pct"] > 0.0
                and cand_pt["clustered_pct"] >= quality_floor
                and all(
                    cand_pt[axis] > b[axis] or cand_pt["clustered_pct"] > b["clustered_pct"]
                    for b in base_pts
                )
            )
            cand_pt[f"strictly_advances_{kind}_pareto"] = strictly_advances

            dominates_any = any(
                cand_pt[axis] >= b[axis] and cand_pt["clustered_pct"] >= b["clustered_pct"]
                and (cand_pt[axis] > b[axis] or cand_pt["clustered_pct"] > b["clustered_pct"])
                for b in base_pts
            )
            cand_pt[f"dominates_any_reference_{kind}"] = dominates_any
            cand_pt[f"dominates_a_baseline_on_{axis}"] = dominates_any

            strongest = max(base_pts, key=lambda b: (b["clustered_pct"], b[axis]))
            cand_pt[f"dominates_strongest_reference_{kind}"] = (
                cand_pt[axis] >= strongest[axis]
                and cand_pt["clustered_pct"] >= strongest["clustered_pct"]
                and (cand_pt[axis] > strongest[axis] or cand_pt["clustered_pct"] > strongest["clustered_pct"])
            )

            if strictly_advances:
                hv_base = hypervolume_2d(base_pts, axis)
                hv_comb = hypervolume_2d(combined, axis)
                cand_pt[f"{kind}_hypervolume_gain_pct"] = (
                    100.0 * (hv_comb - hv_base) / hv_base if hv_base > 0 else (100.0 if hv_comb > 0 else 0.0)
                )
            else:
                cand_pt[f"{kind}_hypervolume_gain_pct"] = 0.0

            faster_or_equal = [b["clustered_pct"] for b in base_pts if b[axis] >= cand_pt[axis]]
            cand_pt[f"gain_pp_at_matched_or_faster_{axis}"] = (
                cand_pt["clustered_pct"] - max(faster_or_equal) if faster_or_equal else None
            )

        best_base_yield = max((b["total_clusters_h"] for b in base_pts), default=None)
        cand_pt["yield_ratio_vs_best_baseline"] = (
            cand_pt["total_clusters_h"] / best_base_yield if best_base_yield and best_base_yield > 0 else None
        )
        track_reports[track] = {
            "candidate": cand_pt,
            "baselines": base_pts,
        }
        all_rows.extend(base_pts)
        all_rows.append(cand_pt)

    cand_list = [v["candidate"] for v in track_reports.values()]
    macro_clustered_pct = fmean(c["clustered_pct"] for c in cand_list)

    # Macro-level paired comparisons against each paired reference arm across the evaluated tracks.
    by_ref_method: dict[str, list[dict]] = {}
    for tr in track_reports.values():
        for b in tr["baselines"]:
            by_ref_method.setdefault(b["method"], []).append(b)
    paired_comparisons = {}
    for ref_label, ref_pts in by_ref_method.items():
        if len(ref_pts) != len(cand_list):
            continue
        ref_macro_pct = fmean(b["clustered_pct"] for b in ref_pts)
        ref_gen_s = math.fsum(3600.0 * b["attempts"] / b["generation_attempts_h"] for b in ref_pts)
        ref_tot_s = math.fsum(3600.0 * b["attempts"] / b["total_attempts_h"] for b in ref_pts)
        ref_yield = fmean(b["total_clusters_h"] for b in ref_pts)
        cand_tot_s = cand_gen_s + cand_af3_s
        q_delta = macro_clustered_pct - ref_macro_pct
        tot_ratio = cand_tot_s / ref_tot_s if ref_tot_s > 0 else math.inf
        gen_ratio = cand_gen_s / ref_gen_s if ref_gen_s > 0 else math.inf
        both_complete = all(c["complete"] for c in cand_list)
        paired_comparisons[ref_label] = {
            "reference_reward": ref_macro_pct / 100.0,
            "reference_macro_clustered_pct": ref_macro_pct,
            "quality_delta_pp": q_delta,
            "generation_cost_ratio": gen_ratio,
            "total_cost_ratio": tot_ratio,
            "yield_delta_clusters_h": fmean(c["total_clusters_h"] for c in cand_list) - ref_yield,
            "dominates_reference_macro": int(
                both_complete and q_delta >= 0.0 and tot_ratio <= 1.0 and (q_delta > 0.0 or tot_ratio < 1.0)
            ),
        }

    return {
        "protocol": PAIRED_TIMING_PROFILE,
        "timing_profile": timing_profile,
        "method": method_name,
        "evaluated_tracks": list(track_reports.keys()),
        "all_four_tracks_evaluated": set(track_reports.keys()) == set(TRACKS.keys()),
        "reward": macro_clustered_pct / 100.0,
        "macro_clustered_pct": macro_clustered_pct,
        "macro_pass_pct": fmean(c["pass_pct"] for c in cand_list),
        "macro_total_clusters_h": fmean(c["total_clusters_h"] for c in cand_list),
        "macro_generation_clusters_h": fmean(c["generation_clusters_h"] for c in cand_list),
        "macro_generation_hypervolume_gain_pct": fmean(c["generation_hypervolume_gain_pct"] for c in cand_list),
        "macro_total_hypervolume_gain_pct": fmean(c["total_hypervolume_gain_pct"] for c in cand_list),
        "generation_seconds": cand_gen_s,
        "af3_seconds": cand_af3_s,
        "total_gpu_seconds": cand_gen_s + cand_af3_s,
        "tracks_on_generation_pareto": sum(int(c["on_generation_pareto"]) for c in cand_list),
        "tracks_on_total_pareto": sum(int(c["on_total_pareto"]) for c in cand_list),
        "tracks_strictly_advancing_generation_pareto": sum(int(c["strictly_advances_generation_pareto"]) for c in cand_list),
        "tracks_strictly_advancing_total_pareto": sum(int(c["strictly_advances_total_pareto"]) for c in cand_list),
        "tracks_dominating_any_reference_generation": sum(int(c["dominates_any_reference_generation"]) for c in cand_list),
        "tracks_dominating_any_reference_total": sum(int(c["dominates_any_reference_total"]) for c in cand_list),
        "tracks_dominating_baseline_total": sum(int(c["dominates_a_baseline_on_total_attempts_h"]) for c in cand_list),
        "tracks_dominating_strongest_reference_total": sum(int(c["dominates_strongest_reference_total"]) for c in cand_list),
        "paired_comparisons": paired_comparisons,
        "by_track": track_reports,
        "plot_rows": all_rows,
    }


def aggregate_repeat_summaries(repeat_summaries: list[dict]) -> dict:
    """Compute each repeat's Pareto summary independently, then aggregate mean and sample SD across repeats."""
    if not repeat_summaries:
        raise ValueError("Expected at least one repeat summary")
    if len(repeat_summaries) == 1:
        out = dict(repeat_summaries[0])
        out["repeats"] = 1
        return out

    first = repeat_summaries[0]
    n_rep = len(repeat_summaries)
    out = dict(first)
    out["repeats"] = n_rep
    scalar_keys = (
        "reward", "macro_clustered_pct", "macro_pass_pct",
        "macro_total_clusters_h", "macro_generation_clusters_h",
        "macro_generation_hypervolume_gain_pct", "macro_total_hypervolume_gain_pct",
        "generation_seconds", "af3_seconds", "total_gpu_seconds",
    )
    for k in scalar_keys:
        vals = [float(s[k]) for s in repeat_summaries]
        out[k] = fmean(vals)
        out[f"{k}_sd"] = stdev(vals)

    out["mean_tracks_strictly_advancing_total_pareto"] = fmean(
        s["tracks_strictly_advancing_total_pareto"] for s in repeat_summaries
    )
    out["mean_tracks_strictly_advancing_generation_pareto"] = fmean(
        s["tracks_strictly_advancing_generation_pareto"] for s in repeat_summaries
    )

    agg_by_track = {}
    agg_plot_rows = []
    pt_numeric_keys = (
        "passes", "clusters", "covered_targets", "pass_pct", "clustered_pct",
        "generation_attempts_h", "total_attempts_h", "generation_clusters_h", "total_clusters_h",
    )
    for track in first["evaluated_tracks"]:
        cand_reps = [s["by_track"][track]["candidate"] for s in repeat_summaries]
        cand_agg = dict(cand_reps[0])
        for k in pt_numeric_keys:
            vals = [float(r[k]) for r in cand_reps]
            cand_agg[k] = fmean(vals)
            cand_agg[f"{k}_sd"] = stdev(vals)
        cand_agg["complete"] = all(bool(r.get("complete", True)) for r in cand_reps)

        base_methods = [b["method"] for b in first["by_track"][track]["baselines"]]
        base_aggs = []
        for idx, mname in enumerate(base_methods):
            breps = [s["by_track"][track]["baselines"][idx] for s in repeat_summaries]
            bagg = dict(breps[0])
            for k in pt_numeric_keys:
                vals = [float(r[k]) for r in breps]
                bagg[k] = fmean(vals)
                bagg[f"{k}_sd"] = stdev(vals)
            bagg["complete"] = all(bool(r.get("complete", True)) for r in breps)
            base_aggs.append(bagg)

        comparable = bool(base_aggs) and cand_agg["complete"] and all(bool(b.get("complete", True)) for b in base_aggs)
        q_floor = min((b["clustered_pct"] for b in base_aggs), default=0.0)
        for kind, axis in (("generation", "generation_attempts_h"), ("total", "total_attempts_h")):
            if not comparable:
                cand_agg[f"on_{kind}_pareto"] = False
                cand_agg[f"strictly_advances_{kind}_pareto"] = False
                cand_agg[f"dominates_any_reference_{kind}"] = False
                cand_agg[f"dominates_a_baseline_on_{axis}"] = False
                cand_agg[f"dominates_strongest_reference_{kind}"] = False
                cand_agg[f"{kind}_hypervolume_gain_pct"] = 0.0
                cand_agg[f"gain_pp_at_matched_or_faster_{axis}"] = None
                continue
            combined = base_aggs + [cand_agg]
            nd = {id(p) for p in frontier(combined, axis)}
            cand_agg[f"on_{kind}_pareto"] = id(cand_agg) in nd
            strictly_adv = (
                cand_agg["clustered_pct"] > 0.0
                and cand_agg["clustered_pct"] >= q_floor
                and all(cand_agg[axis] > b[axis] or cand_agg["clustered_pct"] > b["clustered_pct"] for b in base_aggs)
            )
            cand_agg[f"strictly_advances_{kind}_pareto"] = strictly_adv
            dom_any = any(
                cand_agg[axis] >= b[axis] and cand_agg["clustered_pct"] >= b["clustered_pct"]
                and (cand_agg[axis] > b[axis] or cand_agg["clustered_pct"] > b["clustered_pct"])
                for b in base_aggs
            )
            cand_agg[f"dominates_any_reference_{kind}"] = dom_any
            cand_agg[f"dominates_a_baseline_on_{axis}"] = dom_any
            strongest = max(base_aggs, key=lambda b: (b["clustered_pct"], b[axis]))
            cand_agg[f"dominates_strongest_reference_{kind}"] = (
                cand_agg[axis] >= strongest[axis] and cand_agg["clustered_pct"] >= strongest["clustered_pct"]
                and (cand_agg[axis] > strongest[axis] or cand_agg["clustered_pct"] > strongest["clustered_pct"])
            )
            if strictly_adv:
                hv_base = hypervolume_2d(base_aggs, axis)
                hv_comb = hypervolume_2d(combined, axis)
                cand_agg[f"{kind}_hypervolume_gain_pct"] = (
                    100.0 * (hv_comb - hv_base) / hv_base if hv_base > 0 else 0.0
                )
            else:
                cand_agg[f"{kind}_hypervolume_gain_pct"] = 0.0
            faster = [b["clustered_pct"] for b in base_aggs if b[axis] >= cand_agg[axis]]
            cand_agg[f"gain_pp_at_matched_or_faster_{axis}"] = (
                cand_agg["clustered_pct"] - max(faster) if faster else None
            )

        agg_by_track[track] = {"candidate": cand_agg, "baselines": base_aggs}
        agg_plot_rows.extend(base_aggs)
        agg_plot_rows.append(cand_agg)

    cand_list = [v["candidate"] for v in agg_by_track.values()]
    out["by_track"] = agg_by_track
    out["plot_rows"] = agg_plot_rows
    out["tracks_on_generation_pareto"] = sum(int(c["on_generation_pareto"]) for c in cand_list)
    out["tracks_on_total_pareto"] = sum(int(c["on_total_pareto"]) for c in cand_list)
    out["tracks_strictly_advancing_generation_pareto"] = sum(int(c["strictly_advances_generation_pareto"]) for c in cand_list)
    out["tracks_strictly_advancing_total_pareto"] = sum(int(c["strictly_advances_total_pareto"]) for c in cand_list)
    out["tracks_dominating_any_reference_generation"] = sum(int(c["dominates_any_reference_generation"]) for c in cand_list)
    out["tracks_dominating_any_reference_total"] = sum(int(c["dominates_any_reference_total"]) for c in cand_list)
    out["tracks_dominating_baseline_total"] = sum(int(c["dominates_a_baseline_on_total_attempts_h"]) for c in cand_list)
    out["tracks_dominating_strongest_reference_total"] = sum(int(c["dominates_strongest_reference_total"]) for c in cand_list)

    # Aggregate paired comparisons across repeats
    agg_pc = {}
    for ref_label in first.get("paired_comparisons", {}):
        creps = [s["paired_comparisons"][ref_label] for s in repeat_summaries]
        entry = {}
        for k in ("reference_reward", "reference_macro_clustered_pct", "quality_delta_pp",
                  "generation_cost_ratio", "total_cost_ratio", "yield_delta_clusters_h"):
            vals = [float(r[k]) for r in creps]
            entry[k] = fmean(vals)
            entry[f"{k}_sd"] = stdev(vals)
        entry["dominates_reference_macro"] = int(
            entry["quality_delta_pp"] >= 0.0 and entry["total_cost_ratio"] <= 1.0
            and (entry["quality_delta_pp"] > 0.0 or entry["total_cost_ratio"] < 1.0)
        )
        agg_pc[ref_label] = entry
    out["paired_comparisons"] = agg_pc
    out["per_repeat"] = [
        {k: v for k, v in s.items() if k != "plot_rows"} for s in repeat_summaries
    ]
    return out


def four_track_harbor_metrics(report: dict) -> dict[str, int | float]:
    """Flatten a four-track evaluation report into strictly numeric Harbor 0.23 reward.json metrics."""
    if "invalid_submission" in report:
        return {"reward": 0.0, "invalid_submission": 1}

    def _num(val) -> int | float:
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int):
            return int(val)
        return float(val)

    values: dict[str, int | float] = {"reward": _num(report["reward"])}
    top_keys = (
        "macro_clustered_pct", "macro_pass_pct", "macro_total_clusters_h", "macro_generation_clusters_h",
        "macro_generation_hypervolume_gain_pct", "macro_total_hypervolume_gain_pct",
        "generation_seconds", "af3_seconds", "total_gpu_seconds",
        "attempts_per_target", "seconds_per_slot", "repeats",
        "tracks_on_generation_pareto", "tracks_on_total_pareto",
        "tracks_strictly_advancing_generation_pareto", "tracks_strictly_advancing_total_pareto",
        "tracks_dominating_any_reference_generation", "tracks_dominating_any_reference_total",
        "tracks_dominating_strongest_reference_total",
        "reward_sd", "macro_clustered_pct_sd", "macro_total_clusters_h_sd",
        "mean_tracks_strictly_advancing_total_pareto",
    )
    for k in top_keys:
        if k in report and report[k] is not None and math.isfinite(float(report[k])):
            values[k] = _num(report[k])

    pcs = report.get("paired_comparisons", {})
    if pcs:
        strongest_name = max(pcs, key=lambda n: pcs[n]["reference_macro_clustered_pct"])
        fastest_name = min(pcs, key=lambda n: pcs[n]["total_cost_ratio"])
        for tag, name in (("strongest", strongest_name), ("fastest", fastest_name)):
            comp = pcs[name]
            for ck, cv in comp.items():
                if cv is not None and math.isfinite(float(cv)):
                    values[f"{tag}_{ck}"] = _num(cv)
        for name, comp in pcs.items():
            slug = name.lower().replace("-", "_").replace(" ", "_")
            for ck, cv in comp.items():
                if cv is not None and math.isfinite(float(cv)):
                    values[f"ref_{slug}_{ck}"] = _num(cv)

    for track, tr in report.get("by_track", {}).items():
        short = TRACK_TO_SHORT_KEY.get(track, track.lower().replace(" ", "_"))
        cand = tr["candidate"]
        for ck in (
            "targets", "attempts", "passes", "clusters", "pass_pct", "clustered_pct",
            "generation_attempts_h", "total_attempts_h", "generation_clusters_h", "total_clusters_h",
            "on_generation_pareto", "on_total_pareto",
            "strictly_advances_generation_pareto", "strictly_advances_total_pareto",
            "dominates_any_reference_generation", "dominates_any_reference_total",
            "dominates_strongest_reference_total",
            "generation_hypervolume_gain_pct", "total_hypervolume_gain_pct",
            "clustered_pct_sd", "total_attempts_h_sd", "total_clusters_h_sd",
        ):
            if ck in cand and cand[ck] is not None and math.isfinite(float(cand[ck])):
                values[f"track_{short}_{ck}"] = _num(cand[ck])
        for b in tr["baselines"]:
            bslug = b["method"].lower().replace("-", "_").replace(" ", "_")
            for bk in ("clusters", "clustered_pct", "generation_attempts_h", "total_attempts_h", "total_clusters_h"):
                if bk in b and b[bk] is not None and math.isfinite(float(b[bk])):
                    values[f"track_{short}_ref_{bslug}_{bk}"] = _num(b[bk])

    for arm, acomp in (report.get("archive_200attempt_quality_comparisons") or {}).items():
        aslug = arm.lower().replace("-", "_").replace(" ", "_")
        for ak in ("archive_macro_clustered_pct", "quality_delta_pp", "tracks_beating_archive"):
            if ak in acomp and acomp[ak] is not None and math.isfinite(float(acomp[ak])):
                values[f"archive_200_{aslug}_{ak}"] = _num(acomp[ak])
    return values


def render_paired_pareto_plot(
    points: list[dict],
    path: Path,
    *,
    include_af3: bool,
    evaluated_tracks: list[str],
    attempts_per_target: int,
    repeats: int = 1,
    panel_name: str = "dev",
):
    """Render 4-panel quality-throughput curves labeled with exact target counts, attempt count, and boundary."""
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

    speed = "total_attempts_h" if include_af3 else "generation_attempts_h"
    colors = ["#198572", "#2674aa", "#b06b16", "#803f87"]
    markers = ["o", "s", "^", "D"]
    methods = list(dict.fromkeys(p["method"] for p in points))
    styles = dict(zip(ARMS, zip(colors, markers)))
    extras = [m for m in methods if m not in ARMS]
    for i, method in enumerate(extras):
        styles[method] = ("#c8372d" if i == 0 else plt.get_cmap("tab10")((i + 3) % 10), ["*", "P", "X", "v"][i % 4])

    tracks = [t for t in TRACKS if t in evaluated_tracks]
    if len(tracks) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10.2, 5.8), squeeze=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3), sharex=True)
    fig.subplots_adjust(left=.102, right=.98, bottom=.15, top=.79, wspace=.19, hspace=.38)
    has_paired = any(m in ARMS for m in methods)
    banner = "Paired 4-track quality and throughput" if has_paired else "4-track search quality and throughput"
    fig.text(.064, .956, f"RFD4-Proteina   /   {banner}", fontsize=19, weight="bold", color="#162b36")
    scope = "Generation + shared warm AF3 evaluation" if include_af3 else "Generation/search (per-request subprocess boundary)"
    rep_tag = f"    ·    {repeats} repeats (mean ± SD)" if repeats >= 2 else ""
    fig.text(.066, .913, f"{scope}    ·    {attempts_per_target} attempts/target ({panel_name} panel){rep_tag}", fontsize=10.5, color="#445361")

    handles = [Line2D([], [], linestyle="none", marker=styles[m][1], color=styles[m][0], label=m, markersize=8) for m in methods]
    if has_paired:
        handles.append(Line2D([], [], linestyle=":", color="#687681", label="Observed nondominated curve"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.058, .895), frameon=False,
               ncol=min(6, len(handles)), fontsize=9.5, columnspacing=1.7, handletextpad=.5)
    for unused in list(axes.flat)[len(tracks):]:
        unused.set_visible(False)

    for ax, track in zip(axes.flat, tracks):
        cohort = [p for p in points if p["track"] == track]
        n_tgts = cohort[0]["targets"] if cohort else 0
        edge = sorted(frontier(cohort, speed), key=lambda p: p[speed])
        if edge and len(edge) > 1:
            ax.plot([p[speed] for p in edge], [p["clustered_pct"] for p in edge], ":", linewidth=1.3, color="#939ea4", zorder=1)
        for p in cohort:
            color, marker = styles[p["method"]]
            size = 115 if p["method"] not in ARMS else 75
            ax.scatter(p[speed], p["clustered_pct"], c=[color], marker=marker, s=size, edgecolors="white", linewidths=.9, zorder=3)
            if repeats >= 2 and f"{speed}_sd" in p and "clustered_pct_sd" in p:
                ax.errorbar(
                    p[speed], p["clustered_pct"],
                    xerr=p[f"{speed}_sd"], yerr=p["clustered_pct_sd"],
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5, alpha=0.75, zorder=2,
                )
        ax.set_title(f"{track}  ·  {n_tgts} target{'s' if n_tgts != 1 else ''}", loc="left", fontsize=12, weight="bold", pad=8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
        ax.tick_params(which="minor", bottom=False)
        ymax = max((p["clustered_pct"] + p.get("clustered_pct_sd", 0.0) for p in cohort), default=1.0)
        ax.set_ylim(0, max(1.0, ymax * 1.18))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="both", color="#dde3e6", linewidth=.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a6b1b7")
        ax.tick_params(labelsize=9, length=0, pad=5, colors="#445361")
        ax.set_axisbelow(True)

    fig.supylabel("Clustered pass rate (%)", x=.025, y=.48, fontsize=11)
    fig.supxlabel("Attempts per GPU-hour (log scale)", y=.068, fontsize=11)
    footnote = (
        f"Same-GPU paired evaluation ({PAIRED_TIMING_PROFILE}): candidate and baselines share identical target roster, "
        f"{attempts_per_target} attempts/target, seeds, and timing boundary."
        if has_paired else
        f"Candidate-only evaluation ({PAIRED_TIMING_PROFILE}): {attempts_per_target} attempts/target per-request subprocess "
        f"wall clock plus shared persistent warm AF3 folding."
    )
    fig.text(.066, .021, footnote, fontsize=8.3, color="#54636b")
    fig.savefig(path, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_paired_points_csv(points: list[dict], path: Path):
    """Write paired baseline and candidate operating points with the ordered union of columns."""
    if not points:
        return
    fieldnames = list(dict.fromkeys(k for row in points for k in row.keys()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)


def write_json_atomic(target: Path, payload: dict):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    published = Path(f.name)
    published.chmod(0o644)
    published.replace(target)


def resolve_target_roster(
    panel: str,
    tracks_filter: list[str] | None = None,
    suite_path: Path | None = None,
) -> dict[str, list[str]]:
    if suite_path is not None:
        roster = conditioned_suite_requests(suite_path)
    elif panel == "dev":
        roster = {ds: list(tgts) for ds, tgts in DEV_4TRACK_TARGETS.items()}
    elif panel == "verify":
        roster = {ds: list(tgts) for ds, tgts in verifier_runtime().VERIFY_4TRACK_TARGETS.items()}
    elif panel == "full":
        with (OVERALL / "targets.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r["method"] == "Low"]
        roster = {}
        for r in rows:
            roster.setdefault(r["dataset"], []).append(r["target"])
    else:
        raise ValueError(f"Unknown panel: {panel}")
    if tracks_filter:
        allowed = set(tracks_filter)
        roster = {ds: tgts for ds, tgts in roster.items() if ds in allowed or TRACK_SPECS[ds]["track"] in allowed}
    return roster


def resolve_gpus(spec: str, available: list[dict] | None = None) -> list[dict]:
    """Select one or more dedicated GPUs for target-level parallel paired evaluation."""
    spec = (spec or "auto").strip()
    if spec == "single":
        return [gpu_device()] if available is None else [available[0]]
    pool = visible_gpus() if available is None else list(available)
    if not pool:
        raise RuntimeError("No GPUs available for four-track evaluation")
    if spec == "auto":
        return pool
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    selected = []
    for tok in tokens:
        matched = next((g for g in pool if tok in (str(g.get("index")), g.get("uuid"))), None)
        if matched is None:
            raise ValueError(f"Requested GPU {tok!r} is not among allocated GPUs")
        if matched["uuid"] not in {g["uuid"] for g in selected}:
            selected.append(matched)
    if not selected:
        raise ValueError("Expected at least one GPU in --gpus")
    return selected


def parse_references(raw_refs: list[str] | None) -> tuple[str, ...]:
    if raw_refs is None:
        return ("native-low", "native-high")
    if len(raw_refs) == 1 and raw_refs[0].lower() == "none":
        return ()
    seen = []
    for name in raw_refs:
        if name not in NATIVE_PRESETS:
            raise ValueError(f"Unsupported four-track paired reference {name!r}; choose from {NATIVE_PRESETS} or 'none'")
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "_eval", "_af3_server"), default="run")
    p.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    p.add_argument("--method-name", default="candidate", help="Display label for the evaluated strategy")
    p.add_argument("--panel", choices=("dev", "verify", "full"), default="dev",
                   help="'dev': 12 public targets; 'verify': 16 held-out verifier targets (4 PPI, 5 SM, 4 Enzyme, 3 DNA); 'full': all 141 Figure 2 targets")
    p.add_argument("--suite", type=Path, default=None, help="Optional controller-owned four-track JSON target suite")
    p.add_argument("--tracks", nargs="*", help="Optional subset of datasets or track names for local debugging")
    p.add_argument("--attempts", type=int, default=None, help="Attempts per target (default: 20 for dev, 200 for verify and full)")
    p.add_argument("--repeats", type=int, default=1, help="Number of independent seed repeats for mean ± SD frontier evaluation")
    p.add_argument("--references", nargs="+", default=["native-low", "native-high"],
                   help="Paired executable baselines run on the same GPU and timing boundary (native-low, native-medium, native-high, or 'none')")
    p.add_argument("--gpus", default="auto",
                   help="GPUs for target-level parallel paired evaluation ('auto', 'single', or comma-separated indices/UUIDs)")
    p.add_argument("--seconds-per-slot", type=float, default=25.0,
                   help="Generation/search wall budget per output (default: 25.0 s/slot calibrated for 8xH100 <=4h verification)")
    p.add_argument("--seed-offset", type=int, default=0, help="Additive seed offset for public repeat runs")
    p.add_argument("--af3-seed", type=int, default=727000)
    p.add_argument("--private-seed", type=Path, default=None, help="32-byte private controller seed file for Harbor verification")
    p.add_argument("--private", action="store_true")
    p.add_argument("--assets", type=Path, default=Path("/workspace/data/assets"))
    p.add_argument("--benchmarks-dir", type=Path, default=None, help="Defaults to <assets>/benchmarks")
    p.add_argument("--oracle-exec", default=os.environ.get("ORACLE_EXEC", "/opt/af3env/bin/python"))
    p.add_argument("--af3-sock", type=Path, default=None)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--work", type=Path, required=False)
    p.add_argument("--output", type=Path, required=False, help="Output report directory")
    p.add_argument("--reward-output", type=Path, default=None, help="Optional Harbor flat numeric reward.json path")
    p.add_argument("--details-output", type=Path, default=None, help="Optional Harbor details.json path")
    # Internal worker arguments
    p.add_argument("--dataset", choices=tuple(TRACK_SPECS))
    p.add_argument("--scratch", type=Path)
    p.add_argument("--jax-cache", type=Path)
    p.add_argument("--af3-log", type=Path)
    args = p.parse_args()

    if args.stage == "_eval":
        return _run_eval_worker(args)
    if args.stage == "_af3_server":
        return _run_af3_server(args)

    if args.work is None or args.output is None:
        p.error("--work and --output are required")
    if args.method_name in ARMS:
        p.error(f"--method-name cannot use reserved baseline name {args.method_name!r}")
    if args.attempts is not None and args.attempts < 1:
        p.error("--attempts must be positive")
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if not math.isfinite(args.seconds_per_slot) or args.seconds_per_slot <= 0:
        p.error("--seconds-per-slot must be finite and positive")

    try:
        paired_refs = parse_references(args.references)
    except ValueError as exc:
        p.error(str(exc))

    is_private = bool(args.private or args.private_seed is not None)
    if args.private_seed is not None and args.tracks:
        p.error("Private verification requires all four tracks")

    work = args.work.resolve()
    out_dir = args.output.resolve()
    if out_dir.exists():
        p.error(f"Choose a fresh output directory: {out_dir}")
    work.mkdir(parents=True, exist_ok=True)

    base_seed_offset = args.seed_offset
    af3_seed = args.af3_seed
    suite_digest = None
    if args.private_seed is not None:
        base_seed_offset, af3_seed, suite_digest = verifier_runtime().four_track_seeds(args.private_seed)

    panel_label = "custom_suite" if args.suite is not None else args.panel
    attempts = args.attempts or (200 if panel_label in ("verify", "full") else 20)

    try:
        package = args.submission.resolve()
        if not (package / "infer.py").is_file() or (package / "infer.py").is_symlink():
            raise InvalidPolicy("Submission must provide a regular infer.py entry point across all four tracks")

        assets = args.assets.resolve()
        bench_root = (args.benchmarks_dir or (assets / "benchmarks")).resolve()
        roster = resolve_target_roster(args.panel, args.tracks, suite_path=args.suite)
        sub_sha = package_digest(package)
        gpus = resolve_gpus(args.gpus)
        if args.af3_sock is not None and len(gpus) > 1:
            gpus = gpus[:1]
        device = {
            "name": gpus[0]["name"],
            "uuid": gpus[0]["uuid"],
            "driver": gpus[0]["driver"],
            "device_file": gpus[0]["device_file"],
            "gpu_count": len(gpus),
            "gpu_uuids": [g["uuid"] for g in gpus],
        }

        ref_packages: dict[str, Path] = {}
        ref_digests: dict[str, str] = {}
        for ref_name in paired_refs:
            rdir = package_reference(ref_name, work / "reference_packages" / ref_name)
            ref_packages[ref_name] = rdir
            ref_digests[ref_name] = package_digest(rdir)

        # Build flat list of paired target jobs; candidate + all paired references for each job share one dedicated GPU.
        jobs = []
        for rep_idx in range(args.repeats):
            rep_offset = base_seed_offset + rep_idx * 100_003
            for ds, targets in roster.items():
                spec = TRACK_SPECS[ds]
                sorted_all = sorted(targets)
                for tgt in targets:
                    seed = (spec["seed_base"] + sorted_all.index(tgt) + rep_offset) % (2**31 - 1)
                    jobs.append((rep_idx, ds, spec["track"], tgt, seed))

        job_queue: queue.Queue = queue.Queue()
        for job in jobs:
            job_queue.put(job)

        cand_by_rep = [{TRACK_SPECS[ds]["track"]: {} for ds in roster} for _ in range(args.repeats)]
        cand_complete_by_rep = [{TRACK_SPECS[ds]["track"]: True for ds in roster} for _ in range(args.repeats)]
        refs_by_rep = [{r: {TRACK_SPECS[ds]["track"]: {} for ds in roster} for r in paired_refs} for _ in range(args.repeats)]
        ref_complete_by_rep = [
            {(r, TRACK_SPECS[ds]["track"]): True for r in paired_refs for ds in roster}
            for _ in range(args.repeats)
        ]
        all_trials: list[dict] = []
        state_lock = threading.Lock()
        active_gpus = gpus[: max(1, min(len(gpus), len(jobs)))]

        def _gpu_worker(gpu: dict):
            with managed_af3_server(assets, args.oracle_exec, work, external_sock=args.af3_sock, gpu_info=gpu) as shared_sock:
                while True:
                    try:
                        rep_idx, ds, track, tgt, seed = job_queue.get_nowait()
                    except queue.Empty:
                        break
                    cif_path = bench_root / ds / f"{tgt}.cif.zst"
                    r_work = work / f"repeat_{rep_idx:02d}" / ds / tgt

                    slot_paths, gen_costs = run_conditioned_program(
                        package, assets, r_work / "candidate_gen", cif_path, f"{ds}/{tgt}",
                        attempts, seed, private=is_private, seconds_per_slot=args.seconds_per_slot,
                        gpu_info=gpu,
                    )
                    cand_ok = not gen_costs["timed_out"] and gen_costs["program_exit_code"] == 0
                    m_cand, t_cand = score_target_slots(
                        dataset=ds, target=tgt, method_name=args.method_name, slot_paths=slot_paths,
                        gen_seconds=gen_costs["generation_seconds"], assets=assets, work_dir=r_work / "candidate_eval",
                        af3_seed=af3_seed, oracle_exec=args.oracle_exec, af3_sock=shared_sock,
                        project_root=args.project_root, repeat_index=rep_idx, gpu_info=gpu,
                    )

                    ref_results = {}
                    ref_trials = []
                    for ref_name in paired_refs:
                        ref_label = PAIRED_REFERENCE_LABELS[ref_name]
                        r_slots, r_costs = run_conditioned_program(
                            ref_packages[ref_name], assets, r_work / f"{ref_name}_gen", cif_path, f"{ds}/{tgt}",
                            attempts, seed, private=is_private, seconds_per_slot=args.seconds_per_slot,
                            gpu_info=gpu,
                        )
                        if r_costs["program_exit_code"] != 0 and not r_costs["timed_out"]:
                            raise RuntimeError(f"Trusted paired baseline {ref_name} failed on {ds}/{tgt}")
                        r_ok = not r_costs["timed_out"] and r_costs["program_exit_code"] == 0
                        m_ref, t_ref = score_target_slots(
                            dataset=ds, target=tgt, method_name=ref_label, slot_paths=r_slots,
                            gen_seconds=r_costs["generation_seconds"], assets=assets, work_dir=r_work / f"{ref_name}_eval",
                            af3_seed=af3_seed, oracle_exec=args.oracle_exec, af3_sock=shared_sock,
                            project_root=args.project_root, repeat_index=rep_idx, gpu_info=gpu,
                        )
                        ref_results[ref_name] = (m_ref, r_ok)
                        ref_trials.extend(t_ref)

                    with state_lock:
                        cand_by_rep[rep_idx][track][tgt] = m_cand
                        if not cand_ok:
                            cand_complete_by_rep[rep_idx][track] = False
                        for ref_name, (m_ref, r_ok) in ref_results.items():
                            refs_by_rep[rep_idx][ref_name][track][tgt] = m_ref
                            if not r_ok:
                                ref_complete_by_rep[rep_idx][(ref_name, track)] = False
                        all_trials.extend(t_cand)
                        all_trials.extend(ref_trials)

        if len(active_gpus) == 1:
            _gpu_worker(active_gpus[0])
        else:
            with ThreadPoolExecutor(max_workers=len(active_gpus)) as pool:
                futures = [pool.submit(_gpu_worker, g) for g in active_gpus]
                for fut in futures:
                    fut.result()

        all_trials.sort(key=lambda r: (r["repeat"], r["dataset"], r["target"], r["method"], r["attempt_id"]))

        repeat_summaries = []
        for rep_idx in range(args.repeats):
            paired_base_pts = summarize_paired_baselines(
                refs_by_rep[rep_idx],
                timing_profile=PAIRED_TIMING_PROFILE,
                ref_complete_by_track=ref_complete_by_rep[rep_idx],
            )
            refs_complete_track = {
                TRACK_SPECS[ds]["track"]: all(
                    ref_complete_by_rep[rep_idx][(r, TRACK_SPECS[ds]["track"])] for r in paired_refs
                )
                for ds in roster
            }
            rep_summary = pareto_summary_across_tracks(
                cand_by_rep[rep_idx],
                paired_base_pts,
                method_name=args.method_name,
                timing_profile=PAIRED_TIMING_PROFILE,
                candidate_complete_by_track=cand_complete_by_rep[rep_idx],
                references_complete_by_track=refs_complete_track,
            )
            repeat_summaries.append(rep_summary)

        summary = aggregate_repeat_summaries(repeat_summaries)
        out_dir.mkdir(parents=True, exist_ok=True)
        trial_fields = (
            "repeat", "dataset", "criterion", "method", "target", "attempt_id",
            "scored", "passed", "cluster_id", "generation_gpu_s", "af3_gpu_s",
        )
        trials_path = out_dir / "trials.csv"
        with trials_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trial_fields, lineterminator="\n")
            w.writeheader()
            w.writerows(all_trials)

        archive_by_track = load_archive_context_for_targets(roster)
        archive_q = summarize_archive_200_quality(summary, archive_by_track, attempts_per_target=attempts)
        summary.update({
            "panel": panel_label,
            "attempts_per_target": attempts,
            "seconds_per_slot": args.seconds_per_slot,
            "paired_references": list(paired_refs),
            "reference_sha256": ref_digests,
            "submission_sha256": sub_sha,
            "device": device,
            "generation_timing_boundary": "per_request_infer_subprocess_wall_clock",
            "af3_timing_boundary": "shared_persistent_af3_server_warm_fold",
            "trials_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
            "archive_200attempt_quality_comparisons": archive_q,
            "archive_200attempt_context": {
                "directly_comparable_to_live_run": False,
                "reason": (
                    "Archived reference tables cluster 200 attempts/target under warm batch-amortized native timing "
                    f"({WARM_CAMPAIGN_PROFILE}) or H200 archive timing; live Pareto timing claims use only same-GPU "
                    f"paired baselines at matched attempts={attempts} and boundary {PAIRED_TIMING_PROFILE}. "
                    "When attempts=200, target-matched quality-only deltas appear in archive_200attempt_quality_comparisons."
                ),
                "by_track": archive_by_track,
            },
        })
        if suite_digest is not None:
            summary["suite_sha256"] = suite_digest

        plot_rows = summary.pop("plot_rows")
        if plot_rows:
            write_paired_points_csv(plot_rows, out_dir / "points.csv")
            for fname, inc_af3 in (("pareto_generation.png", False), ("pareto_including_af3.png", True)):
                render_paired_pareto_plot(
                    plot_rows, out_dir / fname, include_af3=inc_af3,
                    evaluated_tracks=summary["evaluated_tracks"],
                    attempts_per_target=attempts, repeats=args.repeats, panel_name=panel_label,
                )
        write_json_atomic(out_dir / "summary.json", summary)

        if args.details_output is not None:
            write_json_atomic(args.details_output.resolve(), summary)
        if args.reward_output is not None:
            write_json_atomic(args.reward_output.resolve(), four_track_harbor_metrics(summary))

        print(json.dumps({
            "submission_sha256": sub_sha,
            "timing_profile": PAIRED_TIMING_PROFILE,
            "panel": panel_label,
            "attempts_per_target": attempts,
            "repeats": args.repeats,
            "paired_references": list(paired_refs),
            "evaluated_tracks": summary["evaluated_tracks"],
            "reward": summary["reward"],
            "macro_clustered_pct": summary["macro_clustered_pct"],
            "macro_total_clusters_h": summary["macro_total_clusters_h"],
            "tracks_strictly_advancing_generation_pareto": summary["tracks_strictly_advancing_generation_pareto"],
            "tracks_strictly_advancing_total_pareto": summary["tracks_strictly_advancing_total_pareto"],
            "tracks_dominating_any_reference_total": summary["tracks_dominating_any_reference_total"],
        }, indent=2, allow_nan=False))

    except InvalidPolicy as exc:
        if args.reward_output is None and args.details_output is None and not is_private:
            raise
        err_details = {"reward": 0.0, "invalid_submission": str(exc)}
        if args.details_output is not None:
            write_json_atomic(args.details_output.resolve(), err_details)
        if args.reward_output is not None:
            write_json_atomic(args.reward_output.resolve(), four_track_harbor_metrics(err_details))
        print(json.dumps(err_details, allow_nan=False))


if __name__ == "__main__":
    main()
