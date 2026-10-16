"""Verify a pinned official tar and safely materialize only per-sample HDF5 files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p = argparse.ArgumentParser()
p.add_argument("--archive", type=Path, required=True)
p.add_argument("--sha256", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with a.archive.open("rb") as f:
    actual = hashlib.file_digest(f, "sha256").hexdigest()
if actual != a.sha256:
    raise ValueError("Source checksum mismatch")
a.output.mkdir(parents=True, exist_ok=True)
names = set()
with tarfile.open(a.archive) as tar:
    for member in tar:
        name = Path(member.name).name
        if (
            not member.isfile()
            or not name.startswith("sample_")
            or not name.endswith(".h5")
        ):
            continue
        if name in names:
            raise ValueError(f"Duplicate source sample: {name}")
        names.add(name)
        dst = a.output / name
        if dst.exists():
            raise ValueError(f"Refusing to overwrite sample: {dst}")
        with (
            tar.extractfile(member) as source,
            dst.with_suffix(".partial").open("wb") as out,
        ):
            shutil.copyfileobj(source, out)
        dst.with_suffix(".partial").rename(dst)
report = {"archive": str(a.archive), "sha256": actual, "samples": len(names)}
(a.output / "source.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
