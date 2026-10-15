# Design system delivery contract

Maintain a complete, relocatable package at `/workspace/submission`:

```text
infer.sh                 # required entry point
harness.json             # what this system is and what it depends on
README.md                # architecture, training or optimisation procedure, limitations
code, checkpoints, and offline assets
setup.sh                 # optional; pinned software only, never model weights
```

The verifier calls `bash infer.sh INPUT_ROOT OUTPUT_ROOT` once, from an unspecified
working directory, with no network, as an unprivileged user, and may relocate the
package first. Input is read-only; output starts empty. Exit zero on a completed
batch and nonzero on an unrecoverable error. Resolve package-owned files relative to
the package; symlinks must not escape it.

## What you deliver

The graded artifact is a **design system**, not necessarily a trained model. Three
paradigms are in scope, declared as `kind` in `harness.json`:

| `kind` | What it is | Analogue |
| --- | --- | --- |
| `trained_generator` | you train a generative model | RFdiffusion3 |
| `predictor_inversion` | you optimise against a pretrained predictor at inference, training nothing | BoltzDesign1 |
| `hybrid` | a trained component plus a pretrained predictor in the loop | - |

```json
{"schema":"protein-design-harness/v1","kind":"hybrid","entry_point":"infer.sh",
 "learned_components":[{"path":"generator.pt","sha256":"3f9c...","parameters":19616907,
                        "trained_in_session":true}],
 "pretrained_assets":[{"asset_id":"boltz1.ckpt","sha256":"a71e...","license":"MIT",
                       "role":"structure predictor inverted at inference"}]}
```

Every declared learned component must exist inside the package with the stated digest.
Every declared pretrained asset must exist in the operator-mounted `/pretrained` store
with the stated digest, a license, and a role. `trained_generator` and `hybrid` require
at least one component with `trained_in_session: true`; `predictor_inversion` and
`hybrid` require at least one declared pretrained asset. There is no parameter limit
and no from-scratch requirement - a system that trains nothing is a legitimate entry.

**Declare pretrained assets rather than shipping them.** The operator mounts them at
the same path in development and verification, which keeps packages small and
provenance auditable. Undeclared pretrained weights inside the package are a contract
violation.

## Inputs

`INPUT_ROOT/manifest.json` declares `schema` (`protein-design-input/v1`), `view`,
per-track `files` with SHA-256, and per-track `counts`. Each `INPUT_ROOT/<track>.jsonl`
holds one request per line with exactly:

```json
{"example_id":"enzyme-0007","track":"enzyme","design_length":118,"num_designs":20,
 "context_path":"contexts/enzyme-0007.npz","context_sha256":"9f3c...",
 "motif_size":3,"hotspot_count":0,"ligand_atom_count":0}
```

`context_path` names an uncompressed `.npz` (no pickled objects) whose arrays share one
coordinate frame:

| Array | Shape | Tracks |
| --- | --- | --- |
| `motif_coords` | `[M, 14, 3]` | `enzyme` |
| `motif_mask` | `[M, 14]` bool | `enzyme` |
| `motif_restypes` | `[M]` text | `enzyme` |
| `ligand_coords` | `[A, 3]` | `sm_binder`, `enzyme` |
| `target_coords` | `[T, 14, 3]` | `ppi_binder` |
| `target_mask` | `[T, 14]` bool | `ppi_binder` |
| `hotspot_indices` | `[H]` | `ppi_binder` |

The tracks are not symmetric. **`sm_binder` supplies a bare ligand and nothing else** -
there is no pocket template, and the design is built around the ligand from scratch.
`enzyme` supplies a catalytic constellation plus its ligand. `motif_mask` marks the
atoms that are actually constrained, so a motif residue may fix only part of its
sidechain.

Structures use the **atom14** layout: up to 14 heavy atoms per residue in a
residue-type-specific order, with backbone `N, CA, C, O` always in slots 0-3. The exact
per-residue ordering is in `baseline/rfd3-small/residues.py`. Hydrogens and `OXT` are excluded.

Request IDs, lengths, counts, and ordering differ between the development and
evaluation suites; do not hard-code them. Any track may ask for any `design_length` in the
range the development splits use (40 to 400 residues), with any motif, ligand or target size
those splits show. Serve every request.

## Outputs

```text
OUTPUT_ROOT/predictions/<track>.jsonl
OUTPUT_ROOT/predictions/structures/<name>.pdb
```

One line per requested `example_id`:

```json
{"example_id":"enzyme-0007","designs":[
  {"design_index":0,"ranking_score":0.91,"structure_path":"enzyme-0007-0.pdb",
   "motif_map":[14,52,93]}]}
```

- `structure_path` is relative to `predictions/structures/` and must stay inside it.
- Each PDB holds **exactly one chain**, exactly `design_length` residues, standard
  residue names only, all four backbone atoms per residue, and finite coordinates.
  Sidechain atoms are optional except where the motif requires them. Every atom must be
  valid for its residue type under the atom14 layout.
- Designs are emitted **in the conditioning frame**. Nothing re-docks them, so a design in
  some other frame is a design that does not engage its ligand or its target: the enzyme
  ligand-clash term reads raw coordinates, and every refold criterion is measured against
  the context where the context sits. Motif RMSD is computed after superposition on the
  motif backbone.
- `motif_map` is required on `enzyme` only: one distinct residue index per motif row,
  in order. The residue at `motif_map[k]` must have the **same residue type** as
  `motif_restypes[k]`, and its sidechain atoms are compared, not just its backbone.
  `sm_binder` has no motif, so it takes no `motif_map`.
- `ranking_score` is finite, larger-is-better, and must not consult ground truth.
- Emit at most `num_designs` designs; extras beyond that count are ignored.

## How designs are scored

**Designability, and nothing besides it.** For every design, in order:

1. **Contract validity** - one chain, exactly `design_length` residues, atom names valid
   for their residue type, finite coordinates, and on `enzyme` a `motif_map` naming one
   distinct in-range residue per motif position whose residue type matches
   `motif_restypes`. This is the contract above rather than a criterion, and motif
   recapitulation has nothing to measure against without it. A design that breaks it
   counts as a failed design.
2. **The one published term measured on the raw design** - `enzyme` only, `success.txt`
   criterion 2: no **non-motif** backbone atom (N, CA or C) within 1.5 A of the closest
   ligand atom. It is a conjunct of the enzyme criterion, checked before folding only
   because a design that fails it cannot pass, so the fold would be wasted.
3. **Designability** - pinned inverse folding designs **4 sequences** at T = 0.1 with
   motif positions held fixed at their declared identities, a pinned structure predictor
   refolds each, and the track's published criterion decides:

   | track | passes when |
   | --- | --- |
   | `unconditional` | backbone RMSD <= 1.5 A |
   | `sm_binder` | backbone RMSD <= 1.5 A, backbone-aligned ligand RMSD <= 5 A, interface PAE <= 1.5, ipTM >= 0.8 |
   | `ppi_binder` | binder backbone RMSD <= 2.0 A and interface min PAE <= 1.5 |
   | `enzyme` | motif **all-atom RMSD <= 1.5 A after aligning on the motif backbone only** (N, CA, C), minimised over symmetric sidechain relabelings (ASP OD1/OD2, GLU OE1/OE2, ARG NH1/NH2, PHE and TYR ring pairs) |

   A design passes if **any** sequence satisfies its criterion, and for `enzyme` any of
   the 5 predictor models. LigandMPNN is used where the context carries a ligand, ProteinMPNN
   otherwise. Aligning the motif on its backbone alone is deliberate: aligning on
   sidechains too would let a design trade sidechain error against backbone error.

   The interface PAE and ipTM terms need a complex-capable predictor. Both come from
   **RoseTTAFold3**, which is what makes them enforceable here.
4. **Novelty against the fit corpus** - a designable design passes only if its maximum TM-score
   (Foldseek 10-941cd33, TM-align mode, normalized by the design's length) to every chain of the fit
   corpus (`development/fit/structures`, the same files mounted at `/workspace/data/fit/structures`) is
   **below 0.8**. Benchmarks report this novelty as the maximum TM-score to the PDB or training set;
   generative models typically sit around 0.45-0.8. A package that retrieved fit-corpus backbones
   reached TM >= 0.9 on 90-100 % of its designs, many identical, which is memorization rather than design.
   Each track also reports `novelty_max_tm_mean` over its designable designs and `designs_novel`.

**No criterion of this task's own invention is applied.** Until 2026-09-26 several were,
all before folding: a geometry envelope (CA-CA continuity, self-clash, radius of
gyration), motif RMSD on the raw design, ligand burial, ppi target clash and hotspot
contact. `success.txt` names none of them, and together they discarded most of the signal
- 59% of designable enzyme designs and every designable ppi binder. Your rotamers in
particular are not judged at all: inverse folding replaces your sequence before anything
is measured, so a backbone-only submission is valid wherever the motif does not require
sidechains.

Two consequences of that, stated rather than buried. **Epitope specificity is not
checked** - with hotspot contact gone, interface PAE proves that *an* interface exists,
not that it is the requested one; RFdiffusion3's own ppi criterion has no hotspot term
either.
And **ligand burial is not checked** - it existed because nothing else forced a design to
engage its ligand, which stopped being true once ipTM and interface PAE were enforced.

Each track reports a **pass rate**: the mean per-target fraction of requested designs
satisfying the criteria above. Every target requests 20 designs (RFdiffusion3's protocol
requests 100; the per-target fraction estimates the same quantity), so a per-target rate is a
fraction of 20. Alongside it each track
reports **diversity**, the fraction of passing designs in distinct within-target clusters
at 2.0 A; rotated or translated duplicates do not count as distinct.

**There is no composite score.** The four tracks are judged by different published
criteria and are not commensurable, and `unconditional` is synthetic to this task rather
than part of RFdiffusion3's benchmark, so any weighted mean would be a number this task
invented rather than one it measured. `reward.json` carries the per-track pass rates,
diversity, and the design counts behind them; compare tracks, not totals.

Diversity is reported, not folded into the pass rate. Mode collapse is invisible in a
pass rate - a single designable fold emitted repeatedly passes every time - so on
`unconditional`, where there is no conditioning to satisfy, diversity is the number that
distinguishes a generator from a lookup table. Read the two together.

## Failure policy

A malformed or invalid **individual design** counts as a failed design, not a void
track: for a generative system, per-sample failure is the measurement. A malformed
**index** - bad schema, unknown or duplicate `example_id` - zeroes that track. Designs
you never emit remain in the denominator, so a missing or empty submission scores zero
rather than scoring nothing. A scoring pipeline that cannot run at all is an
infrastructure error and exits nonzero; it never becomes a zero reward.

`infer.sh` runs once for the whole suite, so a crash or a nonzero exit is an invalid
submission: every track scores 0. A request the package cannot serve must not stop the
batch: write `"designs": []` for that `example_id` (its designs count as failures), log
why, and continue. Before publishing, run

```bash
python /workspace/evaluation/check_package.py --package /workspace/submission
```

It replays the package on every request of `/data/tune` (all four tracks, a spread of design
lengths) with two designs each and applies the verifier's contract checks; it fails if
`infer.sh` exits nonzero or any request lacks a valid design.
