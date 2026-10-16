"""Combine verified shard extractions without copying pixels or losing source IDs."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--sources", nargs="+", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Use a fresh combined-source directory")
files, sources, hashes = {}, [], set()
for directory in a.sources:
    report = json.loads((directory / "source.json").read_text())
    if report["sha256"] in hashes:
        raise ValueError("Duplicate source archive")
    hashes.add(report["sha256"])
    samples = list(directory.glob("sample_*.h5"))
    if len(samples) != report["samples"]:
        raise ValueError("Extraction count differs from its receipt")
    for sample in samples:
        if sample.name in files:
            raise ValueError(f"Source ID collision: {sample.name}")
        files[sample.name] = sample.resolve()
    sources.append(dict(report, directory=str(directory.resolve())))
a.output.mkdir(parents=True)
for name, source in sorted(files.items()):
    (a.output / name).symlink_to(source)
report = {"sources": sources, "samples": len(files), "unique_source_ids": True}
(a.output / "source.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
