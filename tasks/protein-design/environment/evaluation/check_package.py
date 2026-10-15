#!/usr/bin/env python3
"""Replay a package on every development request and check that each one gets a valid design.

    python /workspace/evaluation/check_package.py [--package /workspace/submission]
        [--input /data/tune] [--designs 2]

The verifier calls `infer.sh` once for the whole evaluation suite, and a crash or a nonzero exit
voids every track. This check runs the package the same way on every request of a development
split (all four tracks, design lengths spread over the development range) with a few designs
each. It passes when `infer.sh` exits zero and every request has at least one design that the
verifier's own contract checks accept: one chain, exactly `design_length` residues, valid atom14
records, and for motif tracks a motif map placing the requested residues. It does not score
designability; `score.py` does.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE / "lib", _HERE.parent.parent / "tests"):
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
        break

import grader                                        # noqa: E402
from common import TRACKS                            # noqa: E402
from geometry import parse_allatom_pdb               # noqa: E402


def stage(source, destination, designs):
    """Copy the split's indexes with num_designs lowered; link the contexts."""
    destination.mkdir(parents=True)
    for track in TRACKS:
        index = source / f"{track}.jsonl"
        if not index.is_file():
            continue
        with index.open() as reader, (destination / index.name).open("w") as writer:
            for line in reader:
                request = json.loads(line)
                request["num_designs"] = min(designs, request["num_designs"])
                writer.write(json.dumps(request) + "\n")
    (destination / "contexts").symlink_to((source / "contexts").resolve())


def check(input_root, output_root):
    problems, served, requests = [], 0, 0
    for track in TRACKS:
        index = input_root / f"{track}.jsonl"
        if not index.is_file():
            continue
        answers = {}
        predictions = output_root / "predictions" / f"{track}.jsonl"
        if predictions.is_file():
            for line in predictions.open():
                record = json.loads(line)
                answers[record["example_id"]] = record.get("designs") or []
        for line in index.open():
            request = json.loads(line)
            requests += 1
            name, length = request["example_id"], request["design_length"]
            context = grader.load_context(input_root / request["context_path"])
            reasons = []
            for design in answers.get(name, []):
                try:
                    path = output_root / "predictions" / "structures" / design["structure_path"]
                    structure = parse_allatom_pdb(path, length)
                    reason, _ = grader.validate_request(track, structure, design.get("motif_map"), context)
                except (KeyError, ValueError, OSError) as error:
                    reason = str(error)
                if reason is None:
                    served += 1
                    break
                reasons.append(reason)
            else:
                why = reasons[0] if reasons else "no design emitted"
                problems.append(f"{track} {name} (design_length {length}): {why}")
    return requests, served, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--input", type=Path, default=Path("/data/tune"))
    parser.add_argument("--designs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="check-package-"))
    try:
        input_root, output_root = work / "input", work / "output"
        stage(args.input.resolve(), input_root, args.designs)
        output_root.mkdir()
        run = subprocess.run(["bash", str(args.package.resolve() / "infer.sh"), str(input_root), str(output_root)],
                             cwd=work, timeout=args.timeout)
        if run.returncode != 0:
            print(f"FAIL: infer.sh exited {run.returncode}; the verifier would void every track")
            return 1
        requests, served, problems = check(input_root, output_root)
        for problem in problems[:50]:
            print("unserved:", problem)
        print(f"{served}/{requests} requests have a valid design")
        return 0 if not problems else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
