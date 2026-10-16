# Data: sources, preparation, and layout

Nothing here is redistributed by this repository. The operator prepares everything on
the host from pinned sources.

Two independent halves:

| View | Source | Built by |
| --- | --- | --- |
| **evaluation** | the curated RFdiffusion3 benchmark targets | `ingest_targets.py` |
| **development** (`fit`, `tune`) | a dated RCSB PDB snapshot | `materialize.py` |

## Evaluation: the curated benchmark

The evaluation set is RFdiffusion3's own benchmark, so scores stay comparable to
published numbers. It is **vendored in this repository** at
[`targets/`](targets/) - 1.2 MB, 41 enzyme + 4 small-molecule + 5 binder targets - so
the evaluation view can be built from a clean clone with no external inputs. See
[targets/README.md](targets/README.md) for provenance and attribution.

```bash
export TASK_DATA_DIR=/secure/protein-design

python environment/data/ingest_targets.py \
    --output-root "$TASK_DATA_DIR/evaluation"

# or point at a different target set
python environment/data/ingest_targets.py --results-root /path/to/targets \
    --output-root "$TASK_DATA_DIR/evaluation"
```

That produces 51 targets: 41 `enzyme` (180 residues), 4 `sm_binder` (150), 5
`ppi_binder` (80), and `uncond_L100` (100), the one target synthesised rather than read
from a `.cif`. Each requests **20 designs** - RFdiffusion3's published protocol is
100 backbones per target; the pass rate is a per-target fraction either way - so the view is
1,020 designs and
`ingest_targets.py::DESIGNS_PER_TARGET` matches it, as does `materialize.py`'s copy of the
constant for the development views. Each `.cif` holds only the conditioning - motif
residues and ligand, already in the design frame - and `select_fixed_atoms` names the
atoms actually constrained, which become the context's `motif_mask`. The constraint is
therefore expressed at atom level, exactly as RFdiffusion3 poses it.

Track semantics follow the benchmark, and they are not symmetric:

- `enzyme` - a catalytic constellation (1-7 residues, atom-level) plus its ligand.
- `sm_binder` - **a bare ligand and nothing else.** There is no pocket template; the
  design is built around the ligand from scratch.
- `ppi_binder` - a target chain plus atom-level hotspots.
- `unconditional` - **a design length and nothing else.** The benchmark has no
  unconditional target, so this one is synthesised rather than ingested:
  `ingest_targets.py::UNCONDITIONAL_TARGETS` emits `uncond_L100` at length 100, matching
  the frozen operator definition so the two harnesses agree on the track.

### Contamination

These targets are old PDB entries, so they are in
the training set of every current design model. That is inherent to the benchmark and
is **not** solved here - RFdiffusion3 accepts the same exposure, since scaffolding a
known catalytic site is the task. The temporal holdout below therefore protects only
the development split.

## Development: a dated PDB snapshot

Training data is built directly from the PDB, following RFdiffusion3's dataset
construction (`models/rf3/configs/datasets/train/pdb/*.yaml` in
[RosettaCommons/foundry](https://github.com/RosettaCommons/foundry)): a frozen, dated
entry index plus conditioning derived from coordinates.

Curated annotation files are deliberately **not** used. They lag the PDB by years -
BioLiP2's catalytic annotations stop in 2021, which leaves zero enzyme targets after
that date - and a benchmark that must hold out recent structures cannot depend on a
stale index. The PDB itself is current: 17,156 entries deposited in 2025 alone.

```bash
# archive the pinned bulk indexes
curl -O https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
curl -O https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt

# entry-level filtering and the split, no structure downloads
python environment/data/materialize.py --pdb-index entries.idx \
    --clusters clusters-by-entity-30.txt --plan-only

# full run: downloads assemblies and derives conditioning
python environment/data/materialize.py --pdb-index entries.idx \
    --clusters clusters-by-entity-30.txt \
    --assembly-cache /secure/rcsb-cache \
    --output-root "$TASK_DATA_DIR"
```

Requires `gemmi` and `pyyaml` on the host (`pip install gemmi pyyaml`) - `gemmi` for
every structure read, `pyyaml` for the `ppi` target specification, which is the one of
the three written in real YAML rather than a JSON dialect. `materialize.sh` checks for
both before it does any work. Both scripts record the SHA-256 of every source they
actually used into `provenance.json`, so a prepared dataset states its own provenance
even though the upstream URLs are mutable.

### Download size and time

| what | size | notes |
| --- | ---: | --- |
| `entries.idx` | **55 MiB** | one file, pinned |
| `clusters-by-entity-30.txt` | **21 MiB** | one file, pinned |
| assemblies, `--corpus-targets 16000` (default) | **~3.7 GiB** | extrapolated from a real 9,009-entry cache measured at **2.10 GiB**, i.e. 244 KiB mean |
| **total for a default full run** | **~3.8 GiB** | |

Assemblies are fetched **serially**, one HTTPS GET each, so wall time is latency-bound
rather than bandwidth-bound: measured at **0.18 s per assembly** from a UW host, which
puts a default 16,000-structure run at **roughly one hour**. Expect longer on a slower
link. The cache is content-addressed by PDB id and re-used, so an interrupted run
resumes without re-downloading, and pointing `--assembly-cache` at a shared directory
costs nothing.

`--corpus-targets` scales this linearly: 2,000 structures is ~470 MiB and a few minutes,
which is enough for a smoke run.

This is the offline-rebuild route. If the host has network, `atomworks setup metadata`
downloads upstream's own tables instead (~975 MB archive expanding to ~3.8 GB of
parquet), and the full RCSB divided mmCIF mirror it expects is ~100 GB - see
`starter/TRAINING_DATA.md` §0 for when that is the better trade.

### Filters

Mirroring RFdiffusion3's PDB dataset config:

- `resolution <= 3.5`, method in X-ray / electron microscopy / electron crystallography
- entry must belong to a 30% sequence-identity cluster
- ligands filtered by AlphaFold3's excluded-ligand list (Supp. Table 10), which removes
  crystallisation aids and buffers while **keeping** biologically real cofactors such
  as ZN, MG and FE
- enzymes identified by the PDB `HEADER` classification (the six EC classes), which is
  available in bulk and current, unlike curated catalytic-site annotation

### Splits

Development is split **temporally**. Everything deposited on or after `--eval-cutoff`
(default `2024-01-01`, RFdiffusion3's published training cutoff) is held out;
everything before is training. Cluster disjointness is layered on top: any 30%-identity
cluster represented in the held-out set is dropped from fit and tune entirely.

Deposition date is the conservative field - release always follows deposition, so a
structure deposited after the cutoff provably cannot appear in a model trained before
it. Dates come from `entries.idx`; an entry with no known date cannot be shown to be
safe and is excluded.

Measured against the real index at the default cutoff: 223,883 eligible entries,
29,677 past the cutoff, of which 10,670 carry an enzyme classification.

## Layout

```text
$TASK_DATA_DIR/
  evaluation/                                  # from ingest_targets.py
    <track>.jsonl  contexts/*.npz  manifest.json
  development/
    fit/   <track>.jsonl  contexts/*.npz       # contexts include native_coords
           structures/<xx>/<id>.cif.gz         # the RFD3 training corpus
           index/{pn_units,interfaces}_df_train.parquet
    tune/  <track>.jsonl  contexts/*.npz       # contexts include native_coords
```

Development contexts carry `native_coords [L, 14, 3]`, `native_mask` and
`native_restypes` for training. The evaluation view does not: designability is measured
by refolding, not by comparison to a native, so evaluation needs no reference structure.

## The RFD3 training corpus

The `.npz` contexts are design *requests*. RFdiffusion3 does not train on requests - it
trains on whole structures and samples its own crops and conditioning from them, which
is where its tipatom and unindexed regimes come from. So the fit view also ships the
assemblies themselves plus two index tables that say which example to draw from which
file.

`materialize.py --corpus-targets N` stages the structures, in the RCSB divided layout
(`1a0i` -> `a0/1a0i.cif.gz`) that atomworks' default path template already resolves.
Indexing is a second command, because it needs atomworks while the rest of this
directory needs only gemmi:

```bash
python build_training_index.py \
    --structures  <root>/development/fit/structures \
    --clusters    clusters-by-entity-30.txt \
    --entry-index entries.idx \
    --output      <root>/development/fit/index --workers $(nproc)
```

`--entry-index` is not optional in practice: an `-assembly1.cif.gz` download carries no
entry metadata (its `metadata.id` is the literal `xxxx`, and date and resolution parse as
None), so without it every date filter drops every row. The builder is also shipped into
the image at `/workspace/data/build_training_index.py`, because an agent that wants a
different corpus or a different clustering has to be able to rebuild the tables.

The full schema, the cluster conventions, the offline gotchas and the measured costs are
in `starter/TRAINING_DATA.md`. Rebuilding is only necessary offline or for structures
outside the PDB - with network, `atomworks setup metadata` downloads upstream's own
tables, clusters included; see that page's §0.

## Evaluation assets

Grading needs two pinned models, and **both are public downloads** - there is nothing
for the operator to obtain separately and nothing to mount:

| stage | model | source |
| --- | --- | --- |
| inverse folding | LigandMPNN / ProteinMPNN @ `26ec57ac` | `github.com/dauparas/LigandMPNN`, weights from `files.ipd.uw.edu` |
| structure prediction | RoseTTAFold3 (foundry @ `b02eed6a`) | source vendored with the starter, checkpoint from `files.ipd.uw.edu` |

Both images fetch them at build time; to reproduce the same tree anywhere, run

```bash
bash environment/fetch_designability_weights.sh /assets
```

It pins a sha256 for every file, verifies what is already there, and re-downloads only
what is missing or corrupt, so it is safe to re-run and safe to point at a shared
directory. This replaces the old `PROTEIN_DESIGN_EVAL_ASSETS_DIR` mount, which existed
because AlphaFold 3's parameters cannot be redistributed.

**Running the scorer outside a container needs one more step: the NumPy 2 patch.**
`evaluation/ligandmpnn` vendors upstream byte-identical, and upstream's bundled openfold
predates NumPy 2 - it uses `np.int`, `np.object` and `np.bool`, all removed in NumPy 2,
in six places across three files. Both Dockerfiles `sed` those away after copying, so the
directory stays diffable against upstream; nothing patches the repo copy itself. With
NumPy >= 2 and no patch, `run.py` imports `sc_utils` -> openfold at module scope and
**every** scorer call dies at import on **every** track. Copy the directory and apply the
same substitution the Dockerfiles do:

```bash
cp -r environment/evaluation/ligandmpnn /somewhere/ligandmpnn
sed -i 's/\bnp\.int\b/int/g; s/\bnp\.object\b/object/g; s/\bnp\.bool\b/bool/g' \
    /somewhere/ligandmpnn/openfold/np/{residue_constants.py,relax/utils.py} \
    /somewhere/ligandmpnn/openfold/data/templates.py
export LIGANDMPNN_ROOT=/somewhere/ligandmpnn
```

These are measuring instruments. Pretrained models are permitted in a submission and
must be declared in `harness.json`; the verifier records when a submission carries the
grading checkpoints, because a system optimising against the instrument that scores it
inflates in-silico designability. **That recording matters more now than it used to:**
the scoring predictor is present in the agent's own image, so inverting it no longer
requires the agent to bring its own copy.
