# Curated benchmark targets

The 50 evaluation targets, vendored so the task is reproducible from a clean clone.
`ingest_targets.py` reads this directory by default.

```text
enzyme/  input.yaml  success.txt  M<M-CSA id>_<pdb id>.cif   41 catalytic active sites
sm/      input.yaml  success.txt  <ligand>_...cif             4 small-molecule sites
ppi/     input.yaml  success.txt  <pdb id>_cropped.cif        5 binder targets
```

Total 1.2 MB. These are kept in Git despite the repository's blanket `*.cif`
exclusion - see the negation in `.gitignore` - because they are small and the
evaluation view cannot be built without them.

## What each file is

`input.yaml` is the target specification in RFdiffusion3's dialect: design `length` or
`contig`, the `ligand` CCD codes, and `select_fixed_atoms` naming the atoms that are
actually constrained, per residue. `select_hotspots` appears on binder targets.

Each `.cif` holds **only the conditioning**, already in the design frame - the motif
residues and the ligand, not a whole protein. `M0024_1nzy.cif` is six catalytic
residues (`FAHGWD`) plus its BCA ligand, 52 protein atoms and 57 ligand atoms.
The `sm/` files contain a ligand and no protein at all, which is the point of that
track.

`success.txt` states the per-track success criterion. Those criteria are implemented
verbatim in `tests/pipeline.py`; see `environment/evaluation/RESULTS.md`.

## Provenance and attribution

- Enzyme targets are keyed by **M-CSA** entry id and PDB accession. M-CSA is CC BY 4.0
  (Ribeiro A.J.M. et al., *Nucleic Acids Research*, 2018).
- Coordinates derive from **RCSB PDB** entries, which are CC0.
- Binder targets are the canonical RFdiffusion binder set: insulin receptor (`4zxb`),
  PD-L1 (`5o45`), IL-7Rα (`3di3`), IL-2Rα (`1z92`), TIE2 (`2gy5`).
- The target selections and criteria follow RFdiffusion3
  ([bioRxiv 2025.09.18.676967](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)).

The files were supplied to this task by the task author rather than produced here; the
`_entry.author` field records `rib7`. The underlying sources are permissively licensed
with attribution, but a maintainer should confirm redistribution is intended before
this repository is made public.

## Contamination

These are old PDB entries - `1nzy`, `13pk`, `1dbt` and similar - so they are in the
training set of every current design model. That is inherent to the benchmark and is
not solved here; RFdiffusion3 accepts the same exposure, since scaffolding a known
catalytic site is the task. The temporal holdout in `materialize.py` protects only the
development split.
