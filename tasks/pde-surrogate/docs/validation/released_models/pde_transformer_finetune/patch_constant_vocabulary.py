#!/usr/bin/env python3
"""Append "buoyancy" to upstream's constant vocabulary, idempotently.

PDEArena's one varying physical parameter is a buoyancy force, and
`pdetransformer.data.metadata_remapping.convert_constants` raises
`ValueError: 'buoyancy' is not in list` because its vocabulary has no entry for it.
Appending is what upstream's own note sanctions ("only add new constants to the end
of this list to ensure a consistent encoding"): every existing index keeps its
meaning, and the class embedding table is sized by `num_classes` (1000 rows), not by
the vocabulary, so the released sc-s weights load unchanged and the new index is
simply an untrained row for this fine-tune.

Note that "buoyancy" already exists in the *fields* vocabulary in the same file, so a
naive whole-file membership test reports success while changing nothing. This patch is
scoped to the constants list.

    python3 patch_constant_vocabulary.py [PATH_TO_metadata_remapping.py ...]

With no arguments it patches the installed `pdetransformer`.
"""
import sys
from pathlib import Path

MARKER = "# NOTE: only add new constants to the end of this list to ensure a consistent encoding!"


def patch(path: Path) -> str:
    raw = path.read_bytes().decode()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    head, sep, tail = text.partition("    all_constants = [")
    if not sep:
        return f"{path}: no all_constants list found"
    block, close, rest = tail.partition("    ]")
    if not close:
        return f"{path}: all_constants list is not closed"
    if '"buoyancy"' in block:
        return f"{path}: already patched"
    if block.count(MARKER) != 1:
        return f"{path}: constant-vocabulary marker not found exactly once"
    block = block.replace(MARKER, '"buoyancy",\n        ' + MARKER)
    out = (head + sep + block + close + rest).replace("\n", nl)
    path.write_bytes(out.encode())
    return f"{path}: appended buoyancy"


def main(argv):
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return
    targets = [Path(a) for a in argv]
    if not targets:
        import pdetransformer.data.metadata_remapping as m

        targets = [Path(m.__file__)]
    for t in targets:
        print(patch(t))


if __name__ == "__main__":
    main(sys.argv[1:])
