"""Create an explicitly derived Re4200 binding without editing public assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    receipt = json.loads((data / "materialized.json").read_text())
    base_name = "Cavity_2D_Re4140"
    if receipt["revision"] != "cf5ef6748443ae72e9647f398487e98d421a7243" or base_name not in receipt["verified_directories"]:
        raise RuntimeError("Original pinned cavity assets are not verified")
    original = data / "environments" / base_name
    target = data / "derived/Cavity_2D_Re4200"
    if target.exists():
        raise FileExistsError("Derived resource already exists; do not overwrite")
    properties = (original / "properties_run.toml").read_text()
    properties, count = re.subn(r"(?m)^Re(\s*)=(\s*)4140\s*$", r"Re\1=\g<2>4200", properties)
    if count != 1:
        raise RuntimeError("Expected exactly one source Reynolds property")
    config, count = re.subn(r"(?m)^  Re: 4140$", "  Re: 4200", (original / "environment_config.yaml").read_text())
    if count != 1:
        raise RuntimeError("Expected exactly one source Reynolds YAML field")
    target.mkdir(parents=True)
    for path in original.iterdir():
        if path.name in ("properties_run.toml", "environment_config.yaml"):
            continue
        (target / path.name).symlink_to(os.path.relpath(path, target), target_is_directory=path.is_dir())
    (target / "properties_run.toml").write_text(properties)
    (target / "environment_config.yaml").write_text(config)
    record = {"status": "derived-Re4200-reconstruction-not-original-paper-resource",
        "environment": "Cavity_2D_Re4200", "parent_environment": base_name,
        "parent_revision": receipt["revision"], "reynolds": 4200,
        "changes": {"properties_run.toml": "Re: 4140 -> 4200", "environment_config.yaml": "maia.Re: 4140 -> 4200"},
        "unchanged": ["mesh", "geometry", "boundary conditions", "actuator", "Re4140 restart", "Re4140 published mean"],
        "requirement": "Long uncontrolled relaxation and new target calibration before learning",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_sha256": {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                             for name in ("properties_run.toml", "environment_config.yaml")}}
    (target / "derivation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
