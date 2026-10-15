# Materializing the data (controller side)

What the data are, where they come from and how the conditioning is derived is in
[README.md](README.md); this page is the procedure and the numbers an operator needs.

Both views come out of one script. Run it in the task's development image, which has the
pinned `gemmi` and `pyyaml` the ingest needs:

```bash
cd protein-design
docker build -f environment/Dockerfile -t protein-design-dev .
export TASK_DATA_DIR=/secure/protein-design TASK_PRIVATE_DIR=/secure/protein-design
docker run --rm -v "$PWD":/task:ro -v /secure/pinned-sources:/sources:ro -v "$TASK_DATA_DIR":/out \
  protein-design-dev bash /task/environment/data/materialize.sh /sources /out
```

`materialize.sh SOURCE_DIR [TARGETS_DIR] OUTPUT_ROOT` fails closed when a pinned source is
missing. `SOURCE_DIR` holds the two archived bulk indexes; `TARGETS_DIR` defaults to the
benchmark targets vendored in this repository.

| step | script | output | cost |
|---|---|---|---|
| evaluation view | `ingest_targets.py` | `evaluation/`: 51 targets, `<track>.jsonl`, `contexts/*.npz`, `manifest.json` | seconds; **no download** |
| development corpus | `materialize.py` | `development/{fit,tune}` plus the RFdiffusion3 training assemblies | ~3.8 GiB, about an hour at the default 16,000 structures |
| training index | `build_training_index.py` | the two parquet tables RFdiffusion3 trains from | minutes |

## Prepared copy

A built copy (RCSB snapshot of 2026-10-06, the default 16,000-structure corpus) is on Hugging Face,
so the hour of serial RCSB downloads can be skipped:

```bash
hf download sunweiwei/ai4sci-protein-design-data --repo-type dataset --local-dir dl
for t in dl/development-*.tar; do tar xf "$t" -C "$TASK_DATA_DIR"; done
python environment/data/ingest_targets.py --output-root "$TASK_PRIVATE_DIR/evaluation"
```

`dl/sources-rcsb-cache-*.tar` hold the downloaded assemblies (reusable with `--assembly-cache`),
and `MANIFEST.json` the sha256 of every shard. The evaluation view needs no download.

## Sizes and times

| what | size | notes |
|---|---:|---|
| `entries.idx` | 55 MiB | one file, pinned |
| `clusters-by-entity-30.txt` | 21 MiB | one file, pinned |
| assemblies, `--corpus-targets 16000` (default) | ~3.7 GiB | extrapolated from a measured 9,009-entry cache at 2.10 GiB |
| **default full run** | **~3.8 GiB** | |

Assemblies are fetched serially, one HTTPS GET each, so wall time is latency-bound: measured
at 0.18 s per assembly, which puts a default run at roughly one hour. The cache is
content-addressed by PDB id and re-used, so an interrupted run resumes without
re-downloading, and `--assembly-cache` can point at a shared directory. `--corpus-targets`
scales linearly — 2,000 structures is ~470 MiB and a few minutes, enough for a smoke run.

The evaluation view needs no network at all: its 51 targets are built from material vendored
in this repository.

## Layout and who mounts what

```text
$TASK_DATA_DIR/
  evaluation/                 verifier only (TASK_PRIVATE_DIR): <track>.jsonl, contexts/, manifest.json
  development/fit/            agent (read-only), mounted at /workspace/data/fit
  development/tune/           agent (read-only), mounted at /workspace/data/tune
  development/assemblies/     RFdiffusion3 training corpus and its index parquets
```

`TASK_DATA_DIR` and `TASK_PRIVATE_DIR` are the same root here: the development container
mounts only `development/fit` and `development/tune`, and the verifier mounts only
`evaluation/`. The image links `/data` to `/workspace/data`, so this task's scripts and docs
resolve either path.

## Verifying a build

The evaluation view is deterministic, so a rebuilt view should reproduce the counts and the
per-file digests in its own `manifest.json`:

```bash
python environment/data/ingest_targets.py --output-root /tmp/view
python -c "import json; print(json.load(open('/tmp/view/manifest.json'))['counts'])"
# {'enzyme': 41, 'ppi_binder': 5, 'sm_binder': 4, 'unconditional': 1}
```

The development corpus is not bit-reproducible: it depends on the RCSB snapshot date, so the
cutoff (`2024-01-01` by default) is what fixes it. At that cutoff the real `entries.idx`
gives 223,883 eligible entries, 29,677 past the cutoff, of which 10,670 carry an enzyme
classification.
