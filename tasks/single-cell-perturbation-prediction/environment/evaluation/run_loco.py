#!/usr/bin/env python3
"""Leave-one-context-out development evaluation with the verifier's scoring code.

For each held-out training context C: (optionally) run a training command with the other
contexts, then run the package's infer.sh on train/C/dev_split and score it against
dev_split/responses.h5ad. Writes OUT/<C>/report.json and OUT/summary.json.

    python run_loco.py --package /workspace/baseline/meandelta --outdir /workspace/research/loco/b1 \
        --train-cmd "python build_table.py --data {data} --contexts {train_contexts}"
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cellscore import score_context  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path("/workspace/data"))
    ap.add_argument("--contexts", nargs="+", default=["k562", "jurkat", "hct116", "hek293t"])
    ap.add_argument("--holdout", nargs="*", help="subset of contexts to hold out (default: all)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--train-cmd", help="run in the package dir per fold; {data} {train_contexts} {holdout} are substituted")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    summary = {}
    for held in (a.holdout or a.contexts):
        train = [c for c in a.contexts if c != held]
        fold = a.outdir / held
        fold.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if a.train_cmd:
            cmd = a.train_cmd.format(data=a.data, train_contexts=" ".join(train), holdout=held)
            with (fold / "train.log").open("w") as fh:
                subprocess.run(shlex.split(cmd), cwd=a.package, stdout=fh, stderr=subprocess.STDOUT, check=True)
        split = a.data / "train" / held / "dev_split"
        out = Path(tempfile.mkdtemp(prefix="predictions-", dir=fold))  # on the research disk, not /tmp
        with (fold / "inference.log").open("w") as fh:
            subprocess.run(["bash", str(a.package / "infer.sh"), str(split), str(out)], cwd=a.package,
                           stdout=fh, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "SE600M_DIR": os.environ.get("SE600M_DIR", "/opt/se600m")})
        baseline_dir = split / "noeffect" if (split / "noeffect").exists() else None  # shipped no-effect + real DE cache
        report = score_context(out / "predictions.h5ad", split / "basal.h5ad", split / "responses.h5ad",
                               split / "requests.jsonl", fold / "score", threads=a.threads, baseline_dir=baseline_dir)
        summary[held] = {"score": report["all"]["score"], "normalised": report["all"]["normalised"],
                         "raw": report["all"]["raw"], "seconds": round(time.time() - t0, 1)}
        print(f"[loco] holdout {held}: score {report['all']['score']:.4f} {report['all']['normalised']}")
    summary["mean_score"] = sum(v["score"] for v in summary.values()) / len(summary)
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: (v["score"] if isinstance(v, dict) else v) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
