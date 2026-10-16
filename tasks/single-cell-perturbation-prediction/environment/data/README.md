# Data: release v1

Everything the agent may use is under this directory, mounted read-only; the frozen SE-600M
release is `se600m/`, also reachable at `/opt/se600m`. There is no network. `manifest.json` lists
every training source with URL / pinned revision / size / SHA-256 / licence and the sampling and QC
parameters; `splits.json` describes the split design and counts. Which public screen each test
context comes from is not disclosed: here the test sources
appear only under their context ids (`ctx_near`, `ctx_mid`, `ctx_far`), including in
`gene_panel.csv`. Write derived files under `/workspace/research`.

## Layout

```
manifest.json  splits.json  gene_panel.csv  gene_panel_intersection.json
train/<ctx>/                     ctx in {k562, jurkat, hct116, hek293t}
  cells.h5ad                     raw counts (CSR float32), a seeded per-target sample of cells
  se_embeddings.npy              SE-600M embedding of every row of cells.h5ad (float16, 2058-d)
  pseudobulk.h5ad                one row per target over ALL source cells (never subsampled)
  dev_split/                     leave-one-context-out view with the test shape:
    basal.h5ad  requests.jsonl  context.json  responses.h5ad  manifest.json
test/<ctx>/public/               ctx in {ctx_near, ctx_mid, ctx_far}; identities hidden
  basal.h5ad  requests.jsonl  context.json
priors/                          esm2_gene_embeddings.pt  go_annotations.tsv.gz  depmap_train_lines.csv
se600m/                          se600m_epoch16.ckpt  config.yaml  protein_embeddings.pt  licences
```

Private responses of the test contexts exist only in the verifier's grader container.

## Contexts

| Context | Role | Source | Chemistry | Perturbations in source |
|---|---|---|---|---|
| k562 | train | Replogle 2022 genome-wide (9,866) + essential (2,057) screens, merged; genes = intersection of both panels | 10x 3' v3 | see `manifest.json` |
| jurkat | train | Nadig 2025 essential-scale | 10x 3' v3 | 2,393 |
| hct116 | train | X-Atlas/Orion genome-wide (dual guide) | 10x GEM-X 5' | ~18k |
| hek293t | train | X-Atlas/Orion genome-wide (dual guide) | 10x GEM-X 5' | ~18k |
| ctx_near / ctx_mid | test | two public CRISPRi screens, 3' v3, identities withheld | 10x 3' v3 | 300 requests each |
| ctx_far | test | a public CRISPRi screen on the VCC 2026 assay, identity withheld | 10x Flex | 300 requests |

### Materialised sizes (release v1)

Training contexts, after the per-target cell caps and after the 80 withheld targets are removed:

| Context | Cells | Genes measured | Targets (incl. controls) | Size on disk |
|---|---|---|---|---|
| k562 | 886,308 | 8,202 | 9,791 | 41.2 GB |
| jurkat | 223,109 | 8,882 | 2,314 | 9.0 GB |
| hct116 | 724,427 | 19,092 | 18,214 | 53.0 GB |
| hek293t | 734,336 | 19,092 | 18,232 | 64.3 GB |
| **total** | **2,568,180** | master panel 19,169 | ~38k distinct | **167.5 GB** |

Sizes include `cells.h5ad`, `pseudobulk.h5ad` and `se_embeddings.npy`. On top of that: `priors/`
0.4 GB, `se600m/` 12 GB (fetched separately), `private/` 22.8 GB.

Test contexts (`n_requests` is always 300 = 100 leaderboard + 200 final):

| Context | Tier | Chemistry | Basal cells | Genes measured | Cells per request | Held-out response cells | conserved-strong / context-variable / weak / unmeasured | Targets seen in training / never seen |
|---|---|---|---|---|---|---|---|---|
| ctx_near | near | 10x 3' v3 | 4,976 | 9,624 | 200 | 22,924 | 171 / 118 / 11 / 0 | 220 / 80 |
| ctx_mid | mid | 10x 3' v3 | 10,000 | 8,749 | 200 | 39,693 | 143 / 141 / 16 / 0 | 220 / 80 |
| ctx_far | far | 10x Flex | 10,000 | 18,080 | 400 | 241,887 | 104 / 136 / 55 / 5 | 219 / 81 |

The same numbers are machine-readable in `splits.json` and `manifest.json`.

All contexts are CRISPRi (dCas9-KRAB) knockdowns. Test contexts are tiered by distance from the
training corpus (near: a cancer line with same-protocol siblings in training; mid: a non-cancer
line; far: a different lineage on a different chemistry). Nothing in the shipped files names them.

## Schemas

**`cells.h5ad`** — `X`: raw UMI counts, CSR float32, genes in `gene_panel.csv` order.
`obs`: `target_gene` (canonical symbol, `non-targeting` for controls), `target_ensembl`,
`guide_id`, `batch` (`<screen>:<gem_group>`), `source_screen`, `total_counts`, `pct_counts_mt`,
`is_control`. `var`: index = symbol, `ensembl_id`, `master_index`. `uns`: context, chemistry.

**`pseudobulk.h5ad`** — `obs` index = target (incl. `non-targeting`); `X` = mean over all source
cells of `log1p(CP10k)`; `layers["mean_counts"]`, `layers["delta_log1p_cp10k"]` (minus control
mean), `layers["z_vs_control"]` (delta / (control sd · sqrt(1/n + 1/n_ctrl))); `obs`: `n_cells`,
`knockdown_frac` (1 − target expression ratio; NaN if the target is not in the panel),
`n_sig_genes_z` (|z| > 3), `effect_l2_delta`; `var`: control mean / sd / mean counts.
Targets in the withheld set are absent from every training table and from `cells.h5ad`.

**`dev_split/`** — `basal.h5ad` (≤ 10,000 control cells of the context), `requests.jsonl`
(300 targets with ≥ 40 shipped cells, sampled across effect-size tertiles; `n_cells` = 200),
`responses.h5ad` (the same cells as in `cells.h5ad` for those targets). Same file formats as
the test inputs, so a package can be checked and scored on it unchanged.

**Test `public/`** — `basal.h5ad` (≤ 10,000 control cells; opaque batch labels, opaque cell
names), `requests.jsonl` (300 requests; `n_cells` 200 for 3' v3, 400 for Flex), `context.json`.
The test target lists include 80 withheld genes (never perturbed in training; the same 80 in every test context) and 220 genes
that are perturbed in at least one training context; the report scores both groups.

## Gene vocabulary

`gene_panel.csv` is the master panel (19,167 genes): the 18,080-gene VCC Flex panel first,
in its official order, then every other gene present in a Replogle/Nadig panel. All joins are on
Ensembl IDs; symbols follow the VCC panel, then the 10x 2024-A annotation. Each context stores
its own subset of the panel in master order (`master_index`); `gene_panel_intersection.json`
lists the genes present in every context. Predictions are scored on the test context's own panel.

## Compression (development-side only)

Raw cells per target are capped (K562 genome-wide 64, K562 essential 200, Jurkat 200, Orion
lines 40; controls 20,000 per context), chosen by a seeded draw. Pseudobulk tables and the test
contexts are computed from every source cell. The caps keep the shipped release at about 170 GB
(including SE-600M 12 GB and SE embeddings 10.5 GB), small enough to iterate on.

QC: Replogle/Nadig cells are used as released by the authors (already filtered). Orion cells:
`pass_guide_filter == 1`, `pct_counts_mt ≤ 20`, `total_counts ≥ 2000`. Drop counts are in
`manifest.json`.

## Priors

See `priors/README.md`. DepMap rows are provided only for training lines; the test contexts'
identities are hidden, so a model can only use dependency information it infers from expression.

## Provenance and licences

Replogle 2022 (CC BY 4.0, Figshare 20029387); Nadig 2025 (GEO GSE264667); X-Atlas/Orion
(CC BY-NC-SA 4.0, Hugging Face `Xaira-Therapeutics/X-Atlas-Orion` @ `53a5bc98`); Arc VCC 2025
objects (Arc data terms, pinned GCS generations); SE-600M (`arcinstitute/SE-600M` @ `5a9a80f`,
Arc model licence — read `se600m/MODEL_LICENSE.md`); DepMap 24Q4 (CC BY 4.0); GO (CC BY 4.0).
Exact URLs, sizes and hashes: `manifest.json`.
