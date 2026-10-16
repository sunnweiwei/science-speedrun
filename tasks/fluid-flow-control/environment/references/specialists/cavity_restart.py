"""Inspect and freeze an actual MAIA-generated Re4200 initial-state reference.

Run with the organizer's netCDF4-enabled Python. Does not edit source resources
or rewrite the file header to claim a different physical Reynolds number.
"""

import argparse
import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with Dataset(args.restart) as file:
        values = {"reynolds": float(file.ReynoldsNumber), "mach": float(file.MachNumber),
                  "global_time_step": int(file.globalTimeStep), "physical_time": float(file.time)}
    if values["reynolds"] != 4200 or values["mach"] != .1 or values["global_time_step"] <= 525000:
        raise ValueError("Expected an actually advanced native Re4200/Ma0.1 state")
    record = {"status": "native-derived-Re4200-state-not-original-paper-initial-condition",
              "restart_sha256": hashlib.sha256(args.restart.read_bytes()).hexdigest(), **values,
              "source": str(args.restart.resolve()), "parent_initial_reynolds": 4140,
              "caveat": "This file's finite-time relaxation does not itself establish stationary statistics"}
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
