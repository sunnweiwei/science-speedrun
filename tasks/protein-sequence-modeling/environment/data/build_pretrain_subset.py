#!/usr/bin/env python3
"""Build the decontaminated UniRef50 pretraining subset for protein-sequence-modeling.

Usage:
    build_pretrain_subset.py TARGET_DIRECTORY [--size 1000000] [--uniref50 PATH]

TARGET_DIRECTORY must already hold the converted PEER views (run materialize.sh
first) — the eval sequences are read from them to build the decontamination
query set. Writes:

    <target>/public/development/pretrain/uniref50_subset.fasta
    <target>/public/development/pretrain/manifest.json

Two properties this script exists to enforce, both of which are easy to get
wrong by hand:

1. **uniref50.fasta.gz is sorted by sequence length, descending.** Taking the
   first N records yields a corpus whose median length is ~8,000 residues —
   nothing like the proteins PEER asks about. We reservoir-sample over the whole
   stream with a fixed seed instead.
2. **PEER's held-out proteins are natural sequences that live in UniRef50.**
   Without removal a pretrained model can memorize its own test set. We drop
   every subset sequence within 50% identity of any PEER eval sequence, using
   the same MMseqs2 procedure ESM-2 used to decontaminate its validation split.

Requires: python3, and `mmseqs` on PATH (conda: `conda install -c bioconda
mmseqs2`). The script fails loudly rather than silently skipping the filter.
"""
import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASKS = ("fluorescence", "stability", "beta_lactamase", "solubility",
         "subcellular_localization", "binary_localization", "fold",
         "secondary_structure", "yeast_ppi", "human_ppi", "ppi_affinity")
PAIR_TASKS = {"yeast_ppi", "human_ppi", "ppi_affinity"}
MIN_LENGTH = 30
MAX_LENGTH = 1024
SEED = 20260902
# The UniProt URL below always serves the *current* UniRef release (replaced
# every eight weeks), so the file is pinned here by release, size and SHA-256
# and the build fails closed on drift. Older releases are only available as the
# combined uniref<release>.tar.gz under previous_releases/; see --allow-drift.
PINNED_RELEASE = "2026_03"
PINNED_SIZE = 8_780_552_383
PINNED_SHA256 = "95bdf597d2f892295b5aa2925f5da593e1f2e1fcd7a0a4f5e66b9f1c2e83290d"
PINNED_URL = ("https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/"
              "uniref50.fasta.gz")
PREVIOUS_RELEASES = ("https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
                     f"release-{PINNED_RELEASE}/uniref/uniref{PINNED_RELEASE}.tar.gz")
# ESM-2's own decontamination settings (Lin et al. 2023, Appendix A.1.1).
MMSEQS_ARGS = ["--min-seq-id", "0.5", "--alignment-mode", "3", "-c", "0.8",
               "--cov-mode", "0", "--max-seqs", "300", "-s", "7"]


def verify_pinned_source(source, allow_drift=False):
    """Check the UniRef50 file against the pinned release; fail closed on drift."""
    if not source.is_file():
        return {"sha256": None, "size": None, "matches_pin": False}
    size = source.stat().st_size
    digest = None
    if size == PINNED_SIZE:
        print(f"verifying {source} against pinned UniRef50 {PINNED_RELEASE}", flush=True)
        digest = sha256(source)
    matches = size == PINNED_SIZE and digest == PINNED_SHA256
    if matches:
        print(f"  UniRef50 {PINNED_RELEASE} verified", flush=True)
        return {"sha256": digest, "size": size, "matches_pin": True}
    message = (f"{source} is not the pinned UniRef50 {PINNED_RELEASE} release "
               f"(expected {PINNED_SIZE:,} bytes / sha256 {PINNED_SHA256[:16]}…, "
               f"got {size:,} bytes"
               + (f" / {digest[:16]}…" if digest else "") + ").\n"
               f"UniProt replaces {PINNED_URL} every release; the pinned file is "
               f"inside {PREVIOUS_RELEASES} once it leaves the current slot.\n"
               "Pass --allow-drift to build from this file anyway; the resulting "
               "corpus and any numbers trained on it are not comparable to the "
               "shipped references.")
    if not allow_drift:
        raise SystemExit(message)
    print("WARNING: " + message, file=sys.stderr, flush=True)
    return {"sha256": digest or sha256(source), "size": size, "matches_pin": False}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_fasta(handle):
    identifier, chunks = None, []
    for line in handle:
        if line.startswith(">"):
            if identifier is not None:
                yield identifier, "".join(chunks)
            identifier, chunks = line[1:].split(None, 1)[0], []
        else:
            chunks.append(line.strip())
    if identifier is not None:
        yield identifier, "".join(chunks)


def reservoir_sample(source, size, seed=SEED):
    """Uniform sample of `size` records over a stream of unknown length."""
    rng = random.Random(seed)
    reservoir = []
    seen = considered = 0
    opener = gzip.open if str(source).endswith(".gz") else open
    with opener(source, "rt") as handle:
        for identifier, sequence in read_fasta(handle):
            seen += 1
            if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
                continue
            considered += 1
            if len(reservoir) < size:
                reservoir.append((identifier, sequence))
            else:
                index = rng.randrange(considered)
                if index < size:
                    reservoir[index] = (identifier, sequence)
            if seen % 5_000_000 == 0:
                print(f"  scanned {seen:,} records, {considered:,} in range", flush=True)
    print(f"  scanned {seen:,} records total, {considered:,} within "
          f"[{MIN_LENGTH},{MAX_LENGTH}], sampled {len(reservoir):,}", flush=True)
    return reservoir, seen, considered


def peer_eval_sequences(target):
    """Every sequence in the held-out eval view — the contamination query set."""
    verifier = target / "public" / "verifier"
    sequences = {}
    for task in TASKS:
        path = verifier / f"{task}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run materialize.sh first")
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                features = row["features"]
                parts = ([features["sequence_a"], features["sequence_b"]]
                         if task in PAIR_TASKS else [features["sequence"]])
                for index, sequence in enumerate(parts):
                    sequences[f"{row['example_id']}#{index}"] = sequence
    return sequences


def write_fasta(path, records):
    with path.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start:start + 60] + "\n")


def stream_all(source, destination):
    """Write every in-range sequence to `destination`, without buffering in RAM.

    For `--all` there is nothing to sample, and holding 37M sequences as Python
    strings would cost ~12 GB for no reason.
    """
    seen = kept = residues = 0
    opener = gzip.open if str(source).endswith(".gz") else open
    with opener(source, "rt") as handle, destination.open("w") as out:
        for identifier, sequence in read_fasta(handle):
            seen += 1
            if MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
                kept += 1
                residues += len(sequence)
                out.write(f">{identifier}\n")
                for start in range(0, len(sequence), 60):
                    out.write(sequence[start:start + 60] + "\n")
            if seen % 5_000_000 == 0:
                print(f"  scanned {seen:,} records, {kept:,} in range", flush=True)
    print(f"  scanned {seen:,} records total, {kept:,} within "
          f"[{MIN_LENGTH},{MAX_LENGTH}], {residues:,} residues", flush=True)
    return seen, kept


def filter_fasta(source, destination, drop):
    """Copy `source` to `destination`, omitting records whose id is in `drop`."""
    lengths = []
    with source.open() as handle, destination.open("w") as out:
        for identifier, sequence in read_fasta(handle):
            if identifier in drop:
                continue
            lengths.append(len(sequence))
            out.write(f">{identifier}\n")
            for start in range(0, len(sequence), 60):
                out.write(sequence[start:start + 60] + "\n")
    lengths.sort()
    return lengths


def contaminated_ids(query, eval_sequences, workdir, threads=None):
    """UniRef50 ids in the `query` FASTA matching any PEER eval sequence at >=50%."""
    if shutil.which("mmseqs") is None:
        raise SystemExit(
            "mmseqs not found on PATH. Decontamination is mandatory — without it the "
            "pretraining corpus contains PEER's held-out proteins.\n"
            "Install with: conda install -c bioconda mmseqs2")
    targets = workdir / "peer_eval.fasta"
    write_fasta(targets, sorted(eval_sequences.items()))
    result = workdir / "hits.tsv"
    command = ["mmseqs", "easy-search", str(query), str(targets), str(result),
               str(workdir / "tmp"), *MMSEQS_ARGS,
               "--format-output", "query,target,fident,alnlen"]
    if threads:
        command += ["--threads", str(threads)]
    print("  running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    hits = set()
    with result.open() as handle:
        for line in handle:
            fields = line.split("\t")
            if fields:
                hits.add(fields[0])
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=lambda x: Path(x).resolve())
    parser.add_argument("--size", type=int, default=5_000_000,
                        help="sequences to sample before decontamination (default 5,000,000)")
    parser.add_argument("--all", action="store_true",
                        help="use every in-range sequence instead of sampling "
                             "(the default corpus; streams rather than buffering)")
    parser.add_argument("--uniref50", type=Path,
                        help="local uniref50.fasta[.gz]; downloaded to <target>/cache if omitted")
    parser.add_argument("--release-note", type=Path,
                        help="uniref50.release_note recording the UniRef release")
    parser.add_argument("--workdir", type=Path,
                        help="scratch for candidates + MMseqs temporaries; put this "
                             "on fast node-local storage for the --all build")
    parser.add_argument("--allow-drift", action="store_true",
                        help="build from a uniref50.fasta.gz that is not the pinned "
                             f"{PINNED_RELEASE} release (results are then not "
                             "comparable to the shipped references)")
    parser.add_argument("--threads", type=int,
                        help="threads for the MMseqs2 search (default: mmseqs decides)")
    args = parser.parse_args()

    source = args.uniref50
    if source is None:
        cache = args.target / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        source = cache / "uniref50.fasta.gz"
        if not source.is_file():
            raise SystemExit(
                f"{source} not found. Download it first (~8.2 GiB):\n"
                "  curl -O https://ftp.uniprot.org/pub/databases/uniprot/uniref/"
                "uniref50/uniref50.fasta.gz\n"
                "  curl -O https://ftp.uniprot.org/pub/databases/uniprot/uniref/"
                "uniref50/uniref50.release_note")

    pinned = verify_pinned_source(source, allow_drift=args.allow_drift)

    destination = args.target / "public" / "development" / "pretrain"
    destination.mkdir(parents=True, exist_ok=True)
    corpus = destination / "uniref50_subset.fasta"

    print("collecting PEER eval sequences for decontamination", flush=True)
    eval_sequences = peer_eval_sequences(args.target)
    print(f"  {len(eval_sequences):,} held-out sequences", flush=True)

    scratch = args.workdir or destination.parent
    scratch.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="peer-decontam-", dir=scratch))
    try:
        candidates = workdir / "candidates.fasta"
        if args.all:
            print(f"streaming every in-range sequence from {source}", flush=True)
            scanned, count = stream_all(source, candidates)
            sampling = {"method": "all-in-range", "min_length": MIN_LENGTH,
                        "max_length": MAX_LENGTH, "records_scanned": scanned,
                        "records_in_range": count}
        else:
            print(f"reservoir-sampling {args.size:,} sequences from {source}", flush=True)
            sampled, scanned, in_range = reservoir_sample(source, args.size)
            sampled.sort()
            write_fasta(candidates, sampled)
            del sampled
            sampling = {"method": "reservoir", "seed": SEED, "requested": args.size,
                        "min_length": MIN_LENGTH, "max_length": MAX_LENGTH,
                        "records_scanned": scanned, "records_in_range": in_range}

        removed = contaminated_ids(candidates, eval_sequences, workdir, args.threads)
        print(f"  removed {len(removed):,} contaminated sequences", flush=True)
        lengths = filter_fasta(candidates, corpus, removed)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    residues = sum(lengths)
    kept = lengths
    # The current-release note leaves its version field blank ("Release: , --"),
    # so the source SHA-256 below is the only thing that actually pins a build.
    release = None
    if args.release_note and args.release_note.is_file():
        release = args.release_note.read_text().strip()
    write_json(destination / "manifest.json", {
        "schema": "peer-protein-pretrain/v1",
        "source": "UniRef50",
        "source_release_note": release,
        "source_sha256": pinned["sha256"],
        "source_size": pinned["size"],
        "pinned_release": PINNED_RELEASE if pinned["matches_pin"] else None,
        "sampling": sampling,
        "decontamination": {"tool": "mmseqs easy-search", "args": MMSEQS_ARGS,
                            "query_sequences": len(eval_sequences),
                            "removed": len(removed)},
        "files": {corpus.name: sha256(corpus)},
        "sequences": len(kept),
        "residues": residues,
        "length": {"mean": residues / len(kept) if kept else 0,
                   "median": lengths[len(lengths) // 2] if lengths else 0,
                   "min": lengths[0] if lengths else 0,
                   "max": lengths[-1] if lengths else 0},
    })
    print(f"wrote {len(kept):,} sequences / {residues:,} residues to {corpus}")


if __name__ == "__main__":
    main()
