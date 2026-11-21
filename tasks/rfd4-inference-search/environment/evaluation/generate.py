"""Execute a validated policy with the frozen native sampler on length-only monomers."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from policy import SECONDS_PER_SLOT, cost, native_config


def generate(plan, length, slots, seed, assets, work, *, seconds_per_slot=SECONDS_PER_SLOT):
    work.mkdir(parents=True, exist_ok=False)
    cfg = native_config(plan, length, slots, seed, assets, work / "outputs")
    (work / "monomer_run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = ["rfd4-generate", "--config-dir", str(work), "--config-name", "monomer_run"]
    priorities = {p["step"]: p["priority"] for p in plan["checkpoints"] if "priority" in p}
    if priorities:
        (work / "priorities.json").write_text(json.dumps(priorities))
        cmd[:1] = [sys.executable, str(Path(__file__).with_name("native_worker.py"))]
    if not plan["checkpoints"]:
        cmd.append("~metrics")
    env = dict(os.environ, DATA_PATH=str(assets), HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    t0 = time.monotonic()
    with (work / "generate.log").open("w") as log:
        with subprocess.Popen(cmd, env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                status = process.wait(timeout=seconds_per_slot * slots)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return [], {"generation_seconds": time.monotonic() - t0, "timed_out": True,
                            "reserved_trunk_passes": slots * cost(plan)}
            if status:
                raise subprocess.CalledProcessError(status, cmd)
    elapsed = time.monotonic() - t0
    csvfile = work / "outputs/inference_metrics/per_example_metrics.csv"
    with csvfile.open() as f:
        rows = list(csv.DictReader(f))
    paths = [Path(r["designed_structure_path"]) for r in rows if r["designed_structure_path"]]
    if len(paths) > slots or len(set(paths)) != len(paths) or not all(p.is_file() for p in paths):
        raise RuntimeError("Unexpected native output manifest")
    return sorted(paths), {"generation_seconds": elapsed, "timed_out": False,
                           "reserved_trunk_passes": slots * cost(plan)}
