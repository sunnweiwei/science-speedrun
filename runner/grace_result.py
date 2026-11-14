#!/usr/bin/env python3
"""Separate the package from the end of research from the package after the submission window.

    grace_result.py RESULTS_DIR      (run.sh calls it when the trial ran with GRACE_SECONDS)

Reads RESULTS_DIR/agent/grace-period.json (written inside the container, see
harbor_ext/trial_options.py) and RESULTS_DIR/snapshots/index.tsv. The window is not research
time, so `submission/` must be the package as it stood when research ended:

  unchanged                           the window did not change the package: nothing to do
  split                               `submission/` becomes the snapshot from the end of research,
                                      the package after the window moves to `submission-grace/`
  no_package_at_research_end          nothing was published during research: the only package is
                                      `submission-grace/`, and there is no research result
  research_end_package_not_captured   research ended with a package the collector has no copy of;
                                      the package after the window moves to `submission-grace/`
                                      and there is no research result

Writes the outcome to RESULTS_DIR/grace-period.json and prints the status word.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

EMPTY = ("", "e3b0c44298fc1c14")  # no package directory, or an empty one (digest of no input)


def main(out: Path) -> str:
    record = json.loads((out / "agent" / "grace-period.json").read_text())
    before, after = record["package_digest_at_research_end"], record["package_digest_after_grace"]
    package, window = out / "submission", out / "submission-grace"
    snapshot = None
    index = out / "snapshots" / "index.tsv"
    if index.exists():
        with index.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["digest"] == before and (out / "snapshots" / row["seconds"]).is_dir():
                    snapshot = out / "snapshots" / row["seconds"]  # the latest copy with this digest
    if before == after:
        status = "unchanged"
    elif before in EMPTY:
        status = "no_package_at_research_end"
    elif snapshot is None:
        status = "research_end_package_not_captured"
    else:
        status = "split"
    if status != "unchanged" and package.is_dir() and not window.exists():
        package.rename(window)
        if status == "split":
            shutil.copytree(snapshot, package, symlinks=True)
    record.update(status=status, research_end_snapshot=str(snapshot) if snapshot else None,
                  window_changed_package=before != after)
    (out / "grace-period.json").write_text(json.dumps(record, indent=1) + "\n")
    return status


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
