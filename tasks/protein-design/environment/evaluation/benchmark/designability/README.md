# Real designability: LigandMPNN + RoseTTAFold3

The task's primary metric is **designability**: a design passes when an independent
predictor, folding a sequence that inverse folding put on the design's backbone,
reproduces that backbone (and, for the binder tracks, a confident interface).
`tests/pipeline.py` is the implementation; this directory is a batch driver over it.

`benchmark/` is excluded from the images, so this *driver* is operator-side like the
rest of it. What changed is the thing that matters: **both scoring models now ship in
the images**, so the agent's own `environment/evaluation/score.py` runs the real
criterion - now its only mode - without anything being mounted. This driver exists for
batch runs over a chosen subset, with per-design records.

## Why these two models

| stage | model | where it comes from |
| --- | --- | --- |
| inverse folding | **LigandMPNN / ProteinMPNN** | the ORIGINAL repo, `github.com/dauparas/LigandMPNN` @ `26ec57ac`, cloned by the Dockerfile |
| structure prediction | **RoseTTAFold3** | foundry @ `b02eed6a`, source vendored at `environment/evaluation/rf3` |

Both sets of parameters are public downloads from `files.ipd.uw.edu`, pinned by sha256:

```bash
bash environment/fetch_designability_weights.sh /assets
```

That script is the whole portability story - it is idempotent, verifies every file, and
has nothing site-specific in it, so any cluster can reproduce `/assets` exactly.

### Not foundry's MPNN re-implementation, deliberately

Foundry also ships `models/mpnn`, and using it would be tidier. It is not used, because
upstream says not to - *"please use the old repositories of ProteinMPNN, LigandMPNN, and
SolubleMPNN for model benchmarking/comparison until the API and public weights
stabilize"* - and because that warning checks out. Measured on an enzyme design whose
ligand had **52 of 53 atoms within 5 A of the backbone**, foundry's `ligand_mpnn`
returned **byte-identical sequences and byte-identical confidences with the ligand
present and with it deleted**, reporting `ligand_interface_sequence_recovery=nan`. Its
ligand conditioning is inert, and ligand conditioning is the entire point of the
sm_binder and enzyme tracks. (Confirmed for PDB input; the CIF path was not retested.)

### Why RF3 and not AlphaFold 3

AF3's parameters are not redistributable - they must be received directly from Google
under `WEIGHTS_TERMS_OF_USE.md`. That single fact shaped the task: the scorer could never
be in the agent's image, so an agent could only ever self-score `--geometry-only`, a
validity proxy that then judged generator sidechains this protocol throws away. That is
how tracks with excellent backbones scored a flat 0.0000 with no gradient to improve
against. Both the shortcut and the pre-fold gate behind it were deleted on 2026-09-26:
designability is the whole metric, and the only thing still measured on a raw design is
`success.txt`'s enzyme ligand-clash term.

RF3 removes the constraint, and is a better fit besides:

| | AlphaFold 3 | RoseTTAFold3 |
| --- | --- | --- |
| parameters | non-redistributable, operator-mounted | public download, pinned by sha256 |
| runtime | 8.7 GB apptainer image, `site_packages` outside the .sif, `$HOME` shadowing the container python | plain venv; source vendored, already on `PYTHONPATH` |
| natural target chain | needs an MSA; none mirrored, so the target **did not fold** (CA RMSD 5.3-17.7 A) and every interface number from that run was void | native `template_selection`; **target CA RMSD 0.097 A** |
| ppi protocol | a lab wrapper, plus four failed attempts at hand-rolling a template mmCIF | one field in the input JSON |
| confidences | `iptm`, `chain_pair_pae_min` | the same keys, same AF3-compatible file layout |

**The trade, stated plainly.** The per-track thresholds in `tests/common.py`
(sc_RMSD <= 1.5, ipTM >= 0.8, interface PAE <= 1.5, motif RMSD <= 1.5) come from
RFdiffusion3's published `success.txt`, which was calibrated with AlphaFold. A different
predictor does not preserve their published meaning, and numbers taken here are not
directly comparable with RFdiffusion3's paper. This was a deliberate choice: a criterion
the agent can actually optimise against is worth more to this task than one that matches
a published table but only ever returns zero. Recalibrating the thresholds against RF3 is
open work.

## Running it

```bash
# 0. pick designs. Subsetting is a COMPUTE decision: a full evaluation view is 5,000
#    designs and ~43,000 folds.
python select_designs.py \
    --source unconditional=$RUNS/ep60-uncond/submission/predictions:$DATA/development/tune \
    --source enzyme=$RUNS/ep60-evaluation/submission/predictions:$DATA/evaluation \
    --targets unconditional=20,enzyme=12 \
    --out selection.json

# 1. score. One RF3 call per design; every criterion comes from tests/pipeline.py.
python run_designability.py --selection selection.json --out designability.json
```

Inside the image `--assets /assets` and `--ligandmpnn /opt/ligandmpnn` are the defaults,
so neither flag is normally needed.

## Notes worth keeping

- **RF3 input is its own dialect**, not AlphaFold's: a list of jobs, each `name` plus
  `components` that are `{"seq", "chain_id"}`, `{"ccd_code"}`, `{"smiles"}`,
  `{"path": "...cif"}`. Templates are a separate `template_selection` list of chain
  selections.
- **Batch, or you will spend the run loading weights.** RF3 loads ~2.8 GB per process.
  `predict_all` puts every sequence for a design into ONE call for that reason;
  `predict` is a single-sequence wrapper over it.
- **Read the samples, not the ranked model.** RF3 writes
  `<name>/seed-N_sample-M/<name>_seed-N_sample-M_model.cif` plus one top-ranked
  `<name>_model.cif`. Globbing only the latter silently turns the enzyme track's "any of
  5 predictor models" into "the top-ranked one".
- `chain_pair_pae_min` comes back upper-triangular (AF3's is a full matrix); taking the
  min over off-diagonal non-null entries works for both.
- **Always check the target folded** before believing an interface number. That is what
  invalidated a whole ppi run once.
- LigandMPNN's `run.py` does `from prody import *` at module scope, so `prody` and
  `ml_collections` must be installed or the scorer dies on its first call.

## Bugs in `tests/pipeline.py` that this work surfaced - all fixed

None affected any geometry-only number on record, because that path never touched
`pipeline.py`; all would have fired the first time the real grader ran. That path is gone
now, so `pipeline.py` is on every scoring call there is.

1. **`_motif_recapitulation` made 61% of the enzyme track unpassable by construction.**
   It aligned on motif N/CA/C and returned `None` with fewer than 3 such atoms shared -
   but these contexts carry fixed catalytic **tip** atoms: **25 of the 41 enzyme
   evaluation contexts have fewer than 3 backbone atoms in `motif_mask`** (median 1,
   min 0). `geometry.py::backbone_aligned_motif_rmsd` always had the one-line fallback
   and `pipeline.py` did not, so the two disagreed. Measured: **strict 0/12, with the
   fallback 4/12**, the recovered designs at 0.62-1.21 A against a 1.5 A gate.
2. **`predict` globbed `*_model.cif`**, reducing the enzyme track's 5 models to 1.
3. **`design_sequences` selected `protein_mpnn` but passed only
   `--checkpoint_ligand_mpnn`**; every `--checkpoint_*` default is relative, so it
   resolved against the current directory.
4. **It invoked bare `python`**, inheriting whatever was on PATH.
5. **The ligand HETATM block labelled every atom carbon**, discarding
   `ligand_elements`. Not cosmetic: LigandMPNN embeds het atoms **by element**, so the
   model was conditioned on an all-carbon ligand.

## Measured under the previous (AlphaFold 3) backend

Kept for reference; these are **not** comparable to RF3 numbers. 68 designs, 464 MPNN
sequences, 848 seed-inferences.

| track | designs | pass | best sc_RMSD (median) | limiting criterion |
| --- | ---: | ---: | ---: | --- |
| unconditional | 20 | 7 (35%) | 1.97 A | sc_RMSD <= 1.5 |
| enzyme | 12 | 4 (33%) | 10.55 A | motif RMSD <= 1.5 |
| sm_binder | 16 | 0 | 2.27 A | ipTM max 0.73, interface PAE min 2.46 |
| ppi_binder | 20 | 0 | 1.13 A | interface PAE min 7.04, ipTM max 0.27 |

ppi failing on the interface is what the training recipe predicts: `ppi.frequency: 0.0`,
so upstream's pretraining never trains PPI conditioning at all.
