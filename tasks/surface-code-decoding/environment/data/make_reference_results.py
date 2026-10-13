#!/usr/bin/env python3
"""Write reference/results for the development release: what the reference AlphaQubit model achieved,
without its weights.

    python environment/data/make_reference_results.py --package REFERENCE_SUBMISSION \
        --development DATA/development --output RESULTS [--predictions EXISTING_VALIDATION_PREDICTIONS]

REFERENCE_SUBMISSION is the frozen v1 reference model (operator-only, never mounted for agents). The
script replays it on the label-free validation view, scores it with the validation labels, and adds the
test-split aggregates and the training cost recorded in reproduction/alphaqubit-summary.json. No test
prediction or per-condition test result is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TASK = Path(__file__).resolve().parents[2] / "environment" / "baseline" / "alphaqubit"  # the baseline project
REFERENCE_TEST = {"performance": 0.7550607692307693, "score": 0.8580328368056459,
                  "source": "historical full replay of the frozen v1 reference on all 3,250,000 test shots (README.md)"}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=TASK, env={"PYTHONPATH": str(TASK), "PATH": "/usr/local/bin:/usr/bin:/bin"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="reuse validation predictions of this package instead of replaying")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must not exist")
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "validation-inputs"
        run([sys.executable, "-m", "qecbench.stage_validation", "--source", str(args.development / "validation"), "--output", str(inputs)])
        predictions = args.output / "validation_predictions"
        if args.predictions:
            shutil.copytree(args.predictions, predictions)
        else:
            subprocess.run(["bash", str(args.package / "infer.sh"), str(inputs), str(predictions)], check=True)
        run([sys.executable, "-m", "qecbench.scoring", "--input", str(inputs), "--predictions", str(predictions),
             "--labels", str(args.development / "validation"), "--config", str(TASK / "score_config.json"), "--validation",
             "--output", str(args.output / "validation_report.json")])
    summary = json.loads((TASK / "reproduction/alphaqubit-summary.json").read_text())
    (args.output / "test_summary.json").write_text(json.dumps({**REFERENCE_TEST,
        "reference_lower_bound_mwpm": 0.7160463461538461, "reference_upper_bound_tn": 0.7615159615384615}, indent=2) + "\n")
    (args.output / "training_record.json").write_text(json.dumps({
        "model": "independent AlphaQubit reproduction (not Google code or weights); architecture in reproduction/README.md",
        "recipe": "per group: baselines.alphaqubit_train --stage pretrain (pij simulations) then --stage finetune "
                  "(--init-model pretrain/best.npz); frozen with baselines.alphaqubit_artifact",
        "measured_resources": summary["measured_resources"],
        "architecture": summary.get("architecture"), "limitations": summary.get("limitations")}, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# Reference model results (no weights)\n\n"
        "The reference is the task author's independent AlphaQubit reproduction: ten specialists, pretrained on "
        "pij_from_even_for_odd simulations and fine-tuned on the training rows, selected on validation. Its weights "
        "are not provided; train your own. `validation_predictions/` holds its per-shot probabilities on the validation "
        "split (same schema as a submission's output), `validation_report.json` its validation metrics, "
        "`test_summary.json` its aggregate test performance, and `training_record.json` its recipe and measured "
        "training cost (about 200 GPU-hours on RTX PRO 6000 Blackwell, up to 8 GPUs at once).\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
