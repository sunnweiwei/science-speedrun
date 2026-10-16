# Training data for the vendored RFdiffusion3

The starter is upstream RFdiffusion3 (`starter/rfd3/`, pinned at `b02eed6a`), reduced in
layer width and nothing else. This page is about the one thing that could not be
vendored with it: **the data contract**.

Upstream's own documentation stops short here. `models/rfd3/README.md` devotes two
paragraphs to data - "update the path configs to point to the correct base directories
for the metadata parquets" for the PDB case, and a single sentence for custom datasets
that defers to a URL you cannot reach offline. The page that URL points at is vendored
at `references/atomworks-datasets.md`, and it is genuinely useful: it explains
`PandasDataset`, the loader factories, `ConcatDatasetWithID` and the weighted sampler.
What it does not explain is the *schema* - which columns must exist, what `cluster`
means, how interface clusters are keyed. That is what follows.

**If you have network access, read §0 first.** Upstream publishes the real metadata
tables, and downloading them is both faster and more faithful than rebuilding them.
§1-§6 describe the offline path, which is what the graded image has.

---

## 0. With network: download the real corpus, do not rebuild it

Upstream's metadata is a **public artifact**, which is not obvious from the rfd3 README -
it only says "update the path configs". The download lives in the atomworks CLI, which is
already installed alongside the vendored rfd3:

```sh
# 1. the two metadata tables (~975 MB archive -> ~3.8 GB of parquet)
atomworks setup metadata /data/upstream/metadata

# 2. the structures: RCSB divided mmCIF mirror (~100 GB), or a subset
atomworks pdb sync /data/upstream/pdb
atomworks pdb sync /data/upstream/pdb --pdb-ids-file ids.txt   # subset

# 3. the chemical component dictionary (~2 GB)
atomworks ccd sync /data/upstream/ccd
```

`output_dir` is a **positional argument** on `setup metadata`; the command's own docstring
shows `--output-dir`, which does not exist and will error. Mirroring uses rsync to
`rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/` on **port 33444** (CCD:
`rsync://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd/`), so a site that blocks
outbound rsync can fetch the metadata but not the structures.

Verified 2026-09-22: `https://files.ipd.uw.edu/pub/atomworks/pdb_metadata_latest.tar.gz`,
975,084,126 bytes, last-modified 2025-08-10, sha256
`8f77156c978135bf5fb5cfff8595103c05dccbafa38c85f3e27385869b9db1a0`. It is a `_latest`
URL, so treat that digest as a record of what was checked, not a pin that must hold.
Same host and `/pub/` tree as the scorer weights in
`environment/fetch_designability_weights.sh`.

### Point the configs at it

```yaml
# configs/paths/data/<your>.yaml
pdb_parquet_dir: /data/upstream/metadata
pdb_data_dir:    /data/upstream/pdb
```

and export `PDB_MIRROR_PATH` / `CCD_MIRROR_PATH` for the atomworks side.

**The mirror layout already matches.** `atomworks pdb sync` writes
`a0/1a0i.cif.gz` - characters 2-3 of the id - which is exactly what
`PNUnitsDFParser`'s default `path_template` resolves (§1) and the same layout IPD's own
frozen PDB copy uses. These are plain asymmetric-unit mmCIFs; atomworks builds the
assembly at load time, so **nothing needs pre-expanding** - unlike the offline path,
which stages `-assembly1.cif.gz` files.

**One filename mismatch to fix.** The archive ships `pn_units_df.parquet`, but
`datasets/train/pdb/af3_train_pn_unit.yaml` reads
`${paths.data.pdb_parquet_dir}/pn_units_df_train.parquet`. Symlink it, or override
`dataset.dataset.data`. The interfaces name matches as shipped.

### What you get, and how it differs from a local rebuild

| | downloaded | `build_training_index.py` |
| --- | --- | --- |
| `pn_units_df` rows | 3,813,611 (whole PDB) | whatever you staged |
| `interfaces_df` rows | 6,875,066 | derived |
| required `columns_to_load` | **all present**, both tables | all present |
| `cluster` | upstream's own 40% MMseqs2 hash | 30% RCSB substitute (§3) |
| `q_pn_unit_is_loi` | populated, 7.1% True | always False (§4) |
| entry dates / resolution | in the table | need `entries.idx` (§4) |

So the download **retires every compromise in §3 and §4** for PDB data. Rebuild only for
structures that are not in the PDB.

### The split and the target holdout are filters, not files

The published tables are whole-PDB (max deposition date 2025-06-24). Train/val and the
benchmark holdout are `filters` in the dataset configs, so they are reproducible from the
repo alone. `datasets/train/pdb/rfd3_train_{pn_unit,interface}.yaml` carry

```yaml
- 'pdb_id not in ["7rte", "7m5w", "7n5u"]'
- 'pdb_id not in ["3di3", "5o45", "1z92", "2gy5", "4zxb"]'   # the five ppi eval targets
- "deposition_date < '2024-12-16'"
```

while `af3_train_*.yaml` cut at `2021-09-30`. Applying RFD3's own filters to the
downloaded tables gives:

| cut | pn_units | interfaces | total examples |
| --- | ---: | ---: | ---: |
| `< 2021-09-30` | 2,395,525 | 4,045,470 | **6,440,995** |
| `< 2024-12-16` | 3,258,883 | 5,632,469 | **8,891,352** |

**Beware a trap if you compare against runs on IPD paths.** Their `pn_units_df_train.parquet`
already has the 2021 cut baked in (max date 2021-09-20), so `rfd3_train_*`'s
`< '2024-12-16'` is a **no-op there** - those runs saw a 2021 corpus. Use the 2021 cut to
reproduce them, and expect a ~10% superset anyway from the newer snapshot.

Cluster **labels** are not portable across snapshots: on the 2.41M example ids shared with
the IPD tables the string matches for 86%, the rest being hash relabelings as new members
join a cluster. The partition is the same scheme; the ids are not join keys.

---

## 1. What RFD3 actually trains on

Not design requests. The four per-track `.npz` contexts under `/data/*/contexts/` are
inference inputs - a length, a motif, a ligand. Upstream trains on **whole structures**
and samples its own crops and conditioning from them. That is where the tipatom regime
(53% of pretraining) and the unindexed regime come from: they are re-rolled on every
draw, not stored.

So training needs two things:

| what | where | who makes it |
| --- | --- | --- |
| mmCIF assemblies | `/data/development/fit/structures/<xx>/<id>.cif.gz` | operator (`materialize.py --corpus-targets`) |
| two index tables | `/data/development/fit/index/*.parquet` | `data/build_training_index.py` |

The `<xx>` shard is characters 2-3 of the PDB id (`1a0i` -> `a0/1a0i.cif.gz`). That is
the RCSB divided layout, which is what `PNUnitsDFParser`'s default `path_template`
(`{base_dir}/{pdb_id[1:3]}/{pdb_id}{file_extension}`) already resolves - so no path
override is needed.

**An example is a row, not a file.** `pn_units_df_train.parquet` has one row per PN unit
(one polymer or non-polymer unit of one assembly); `interfaces_df_train.parquet` has one
row per contacting pair. A single entry therefore contributes several examples.

---

## 2. The schema

Three consumers fix it, and every column below exists because one of them reads it.

### `pn_units_df_train.parquet`

| column | dtype | meaning |
| --- | --- | --- |
| `example_id` | str | `{['pdb', 'pn_units']}{1owi}{1}{['A_1']}` - build it with `atomworks.ml.example_id.generate_example_id`, never by hand |
| `pdb_id` | str | lower-case, resolves the file through the path template |
| `assembly_id` | str | which assembly to build |
| `deposition_date` | datetime64 | read by the `filters` |
| `resolution` | float | read by the `filters`; NaN is allowed and passes `< 9.0` as False |
| `num_polymer_pn_units` | int | read by the `filters` (`<= 300`) |
| `method` | str | `X-RAY_DIFFRACTION`, underscored |
| `cluster` | str | see §3 - the sampler's homology balance |
| `n_prot`, `n_nuc`, `n_ligand`, `n_peptide` | int | AF-3 composition weights; exactly one is 1 |
| `total_num_atoms_in_unprocessed_assembly` | int | loaded by the config, not filtered on |
| `q_pn_unit_iid` | str | **the query unit - this is what the crop is centred on** |
| `q_pn_unit_non_polymer_res_names` | str or None | CCD codes; the AF-3 excluded-ligand filter matches on it |
| `all_pn_unit_iids_after_processing` | JSON str | which units survive clash resolution |
| `q_pn_unit_is_loi` | bool | ligand of interest; see §4 |

### `interfaces_df_train.parquet`

Same common columns, then `pn_unit_1_iid` / `pn_unit_2_iid` (both query units - the crop
is centred on the interface), `pn_unit_1_non_polymer_res_names` /
`pn_unit_2_non_polymer_res_names`, `is_inter_molecule` (bool, filtered on) and
`involves_loi`. The composition counts are summed over **both** units, so a
protein-ligand interface is `n_prot=1, n_ligand=1`.

The consumers:

* `atomworks.ml.datasets.parsers.PNUnitsDFParser` / `InterfacesDFParser` - find the file,
  pick the assembly, choose the crop centre.
* `atomworks.ml.samplers.calculate_af3_example_weights` - **asserts** on
  `["n_prot", "n_nuc", "n_ligand", "cluster_size"]`. `cluster_size` is not a column you
  write: the sampler derives it from `cluster` and asserts it is never NaN.
* the dataset configs' `filters` - `deposition_date`, `resolution`,
  `num_polymer_pn_units`, `cluster.notnull()`, the `*_non_polymer_res_names` regex, and
  `is_inter_molecule` for interfaces.

### How the weights use it

```
w  ∝  (β / N_cluster) · (a_prot·n_prot + a_peptide·n_peptide + a_nuc·n_nuc
                         + a_ligand·n_ligand + a_loi·is_loi)
```

with `beta: 0.5` and `a_prot 3.0 / a_nuc 0.0 / a_ligand 1.0 / a_loi 5.0`
(`datasets/train/pdb/base.yaml`). The two tables are concatenated, their weight vectors
are concatenated with them, and one `WeightedRandomSampler(..., replacement=True)`
covers the whole thing. There is **no fixed interface:pn_unit ratio** - the mix falls
out of the weights.

Two consequences worth internalising:

1. **Sampling is with replacement and an "epoch" is just a counter**
   (`n_examples_per_epoch`). Nothing is exhausted, so a small corpus needs no structural
   change - it is revisited sooner, and each revisit re-rolls the crop and the
   conditioning, so a repeat is not the same example.
2. **`cluster` is the only thing clustering feeds**, and it enters as a scalar divisor.
   Without it, `datasets/train/pdb/base_no_weights.yaml` gives uniform sampling.
   Training still runs; you lose homology balancing.

---

## 3. The cluster conventions

Nothing documents these and nothing errors if you get them wrong - the weights just come
out wrong, silently. All three were established by reading upstream's parquets.

1. **Polymers are keyed by a clustering, not by the entity.** Upstream uses a
   representative-sequence hash from a 40%-identity MMseqs2 clustering
   (`3ad9e8af6a4`). This task substitutes **RCSB's 30% entity clusters**
   (`rcsb30:57016`), because that is a public, digest-pinned input the task already
   downloads *and* it is the partition the fit/evaluation split is made disjoint on -
   so the sampler's balancing and the split agree instead of fighting. Where an entity
   is missing from the cluster file we fall back to atomworks' exact-sequence hash,
   which under-clusters but never collides.
2. **Non-polymers are keyed by CCD code.** A bare ligand unit's cluster is literally
   `426` or `HEM`. So all heme sites in the PDB are one cluster and are damped together.
3. **Interface clusters are the two sides joined by `+`** - `2eeec33883b+SPM`,
   `HEM+rcsb30:57016`. Sorted, so the key does not depend on which side was listed
   first. Get this wrong and interface homology silently collapses to single-chain
   clustering.

Why MMseqs2 is not an option here: it is not in the image, `PIP_NO_INDEX=1` and no
network mean it cannot be installed, and clustering ~16k entries by pairwise alignment
does not fit the budget. The RCSB clusters are already a task input
(`data/sources.json`, `rcsb_clusters_30`), pinned by digest.

---

## 4. Building the index offline

This is the path for the graded image, which has no network, and for structures that are
not in the PDB. **For PDB data with network access, §0 is both cheaper and more faithful.**

```sh
python /workspace/data/build_training_index.py \
    --structures /data/development/fit/structures \
    --clusters   /data/sources/clusters-by-entity-30.txt \
    --entry-index /data/sources/entries.idx \
    --output     /data/development/fit/index \
    --workers    $(nproc)
```

Four things that will bite if you write your own instead:

1. **`DataPreprocessor(from_rcsb=False)` is mandatory offline.** The default
   (`from_rcsb=True`) POSTs to `data.rcsb.org/graphql` for ligand validity scores with
   **no try/except around the request**, so it raises `ConnectionError` rather than
   degrading. The cost of `False` is that `is_loi` is False everywhere, which switches
   off the `a_loi: 5.0` term - about 2.9% of upstream's rows carry it.
2. **An `-assembly1.cif.gz` download carries no entry metadata.** Its `metadata.id` is
   the literal `xxxx`, and deposition date, release date and resolution all parse as
   None. They have to come from `entries.idx`, which is why `--entry-index` exists. Skip
   it and every date filter drops every row.
3. **Give each structure a timeout.** Cost is not proportional to file size: `4nlm` is
   175 KB and ran for over ten minutes, while the median entry is ~2 s. The builder
   budgets 120 s per structure and records the skips in `index_report.json`.
4. **Cast the PN unit ids to `str`.** atomworks returns numpy `str_`, and
   `generate_example_id` formats the list with `repr`, so a numpy scalar is written into
   the id as `np.str_('A_1')` - and `parse_example_id`, which `eval`s that field, then
   fails on it.

To build a corpus of your own from scratch, the preprocessing half is
`atomworks.ml.preprocessing.get_pn_unit_data_from_structure` - pure Python over biotite,
no binaries, no network. It gives you per-unit records (sequences, sequence hashes,
contacting units, residue counts, clash severity); the cluster join and the interface
derivation are what `build_training_index.py` adds on top.

### Measured cost

On one core, preprocessing runs at **~2 s per structure** (median; mean is higher
because of a long tail). The transform pipeline that runs at *training* time - parse,
assembly build, crop, condition sampling, featurisation - costs a **median 2.05 s and
mean 2.50 s per drawn example** on one core, measured through upstream's own loader at
crop 256 with 16 noise draws.

That sets the shape of the problem: eight
loader workers deliver roughly 3 examples/s, so the CPUs are not obviously the binding
constraint, and the honest answer is that **you should measure GPU utilisation before
assuming either side is**. Turning on `save_to_cache` for the parsed-CIF cache
(`paths.data.cif_cache_dir`, see `configs/paths/data/task.yaml`) moves parse cost off
the critical path on every revisit, which matters because sampling is with replacement.

---

## 5. Training

```sh
python /workspace/baseline/rfd3-small/rfd3/models/rfd3/src/rfd3/train.py experiment=pretrain_small_ppi
```

`pretrain_small_ppi` is upstream's recipe with the widths reduced and PPI conditioning
on, and it already points at this container's data by default - there is nothing to
override. It changes no other recipe leaf: 18 diffusion blocks, the condition mix
(tipatom 5.0 / unconditional 2.0 / island 2.0 / sequence_design 0.5), the unindexed
regime at `p_unindex_motif_tokens: 0.5` on island and tipatom, crop 256, the AF-3
optimizer, schedule and EMA, and every loss weight.

What it does change, and why:

| change | reason |
| --- | --- |
| `paths/data: task` | upstream's defaults are IPD mounts that do not exist here |
| parquet filenames + date filter | match what the index builder writes and the task's own split boundary |
| `datasets.val: null` | upstream's validation sets read design-benchmark JSON that is not redistributed; a missing file fails at dataset construction. Score with `evaluation/score.py` on the tune view instead |
| `calculate_hbonds: 0` | HBPLUS is not installed, and 0.2 raises at runtime. It is a free download - install it and restore 0.2 to get the channel back |
| `grad_accum_steps: 3`, `devices_per_node: 1` | one GPU, so upstream's 3-structures-per-optimizer-step has to come from accumulation. Keeping it is what makes the AF-3 warmup and decay mean the same thing per unit of data |

### Verified end to end

Run 2026-09-17 on one RTX A4000, from an index this pipeline built over task-cache
assemblies. `train.py` on this experiment reached 2 optimizer steps over 6 batches and
saved a checkpoint, with:

* **12,445,554 trainable parameters** - identical to the reference reduced run;
* examples drawn from **both** tables (`pdb/pn_units` 66.7%, `pdb/interfaces` 33.3%) by
  the weighted sampler, e.g. `{['pdb', 'interfaces']}{1a31}{1}{['B_1', 'C_1']}`;
* `Mse Loss Mean` **1.0126** at step 0, which is what an EDM-preconditioned loss should
  read for an untrained network - 1.0 is exactly a zero predictor;
* **`lp_norm_unindexed_diffused` 0.6489, nonzero** - the unindexed regime is live and
  contributing gradient, which is the whole reason for training upstream rather than the
  port;
* validation correctly skipped ("No validation datasets provided").

Diffing the checkpoint's own resolved `train_cfg` against a fresh
`compose(experiment=pretrain)` leaves **three** unexplained differing leaves once width
cuts, data re-pointing, removed validation and the smoke run's own `max_epochs` /
`n_examples_per_epoch` / `diffusion_batch_size_train` are set aside: two dataloader knobs
passed on the command line, and `calculate_hbonds` (§5). Crop 256, `max_atoms_in_crop`
2560, `grad_accum_steps` 3, 18 diffusion blocks, tipatom frequency 5.0 and
`p_unindex_motif_tokens` 0.5 on both island and tipatom all match upstream exactly.

### Known departures from upstream, all deliberate

* **30% RCSB clusters instead of 40% MMseqs2** (§3) - a different, stricter partition.
  Avoidable with network: the downloaded tables carry upstream's own clusters (§0).
* **`is_loi` always False** (§4) - the `a_loi` term never fires. Also avoidable via §0,
  where the column is populated.
* **No hydrogen-bond conditioning** - HBPLUS absent.
* **No MACE embedding cache** - `residue_cache_dir` starts empty and atomworks falls
  back to RDKit reference conformers. Upstream hits the same fallback on this cluster
  for any component its shared cache lacks.
* **No validation loop** - the task's grader is the metric.

---

## 6. From checkpoint to submission

`rfd3_infer.sh` closes the loop:

```sh
# 100 designs per target by default (N_BATCHES=25 x 4 diffusion samples); lower
# N_BATCHES for a quicker, noisier read.
bash rfd3_infer.sh <ckpt> /data/tune /workspace/research/run
python /workspace/evaluation/score.py --input /data/tune \
    --designs /workspace/research/run/submission --tracks enzyme
```

Three stages, with upstream's inference untouched in the middle:

| stage | what |
| --- | --- |
| `rfd3_inputs.py` | request `.npz` -> `InputSpecification` + a conditioning structure rebuilt from the context |
| `rfd3 design` | upstream, unmodified. Reduced widths come from the checkpoint's own `train_cfg` |
| `rfd3_collect.py` | output mmCIF -> one PDB per design plus `<track>.jsonl`, with `motif_map` recovered for enzyme |

`rfd3_inputs.py` works on **fit and tune**, which is what an agent has - the container
mounts only `development`. It rebuilds the conditioning rather than reusing upstream's
published benchmark specs, which exist only for the evaluation view the verifier holds
(that path is `environment/evaluation/benchmark/` in the repo, operator-side). All 160 tune specs across the
four tracks pass upstream's own CPU-only `DesignInputSpecification.safe_init`.

Two details worth knowing before you modify it:

* **Ligand atom names are reconstructed.** A context stores ligand coordinates and
  elements but no names, and RFD3 needs names that match the chemical component or bonds
  and the reference conformer will not resolve. Materialization keeps deposition order,
  which follows the CCD, so the observed element sequence is embedded into the CCD's
  heavy-atom sequence, earliest match first. 22 of 24 sampled ligands match
  element-for-element; the other two have *fewer* atoms than the CCD (unresolved
  density), which is exactly why a subsequence embedding is used rather than positional
  assignment.
* **Enzyme goes through `unindex`, not a contig.** The request gives motif geometry with
  no positions, so the model chooses placement and `motif_map` reports it back. That is
  upstream's guidepost setting and the reason for training upstream at all.

Three bugs this path already cost, kept here so they are not rediscovered:

1. A pinned contig of `150` parses as **residue index 150**, not a 150-residue design.
   Write `150-150`.
2. `diffused_index_map` keys come back **truncated** (`B1029` -> `B102`), so exact lookup
   silently drops designs; match on unique prefix.
3. The coordinate fallback must not demand all four backbone atoms - an enzyme context
   carries only 2-3 fixed tip atoms per motif residue, so a four-atom requirement can
   never fire.
