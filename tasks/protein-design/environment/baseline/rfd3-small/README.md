# RFdiffusion3 starter

**Upstream RFdiffusion3 itself**, vendored at `rfd3/` at the revision `task.toml` pins
(`b02eed6a`), reduced in layer width and nothing else. Its dependencies are installed in
the image. This is a **baseline to replace**, not a solution: it establishes the
interface, the recipe and the conditioning, and leaves the science open.

The starter is one of the three permitted paradigms, a `trained_generator`. A
`predictor_inversion` system that trains nothing is equally valid and would replace this
directory entirely.

```bash
# train (agent scale: 1 GPU, data staged into the image - read TRAINING_DATA.md first)
python rfd3/models/rfd3/src/rfd3/train.py experiment=pretrain_small_ppi

# design from a checkpoint, in the task's submission format. The default is 25 batches of
# 4 diffusion samples = 100 designs per target, which is what every view requests and what
# the reference row in ../evaluation/RESULTS.md was produced at.
bash rfd3_infer.sh <ckpt> /data/tune /workspace/research/run

# ...or fewer designs per target for a quick, higher-variance read while iterating.
# N_BATCHES x 4 = designs per target.
N_BATCHES=2 bash rfd3_infer.sh <ckpt> /data/tune /workspace/research/run

# the real metric - LigandMPNN + RoseTTAFold3, both already in this image
python /workspace/evaluation/score.py --input /data/tune \
    --designs /workspace/research/run/submission --tracks enzyme
```

**Designability is the only score, and it runs here.** The parameters are at `/assets` and
nothing has to be mounted or downloaded. There is no geometry-only shortcut any more: it
was deleted on 2026-09-26 along with the pre-fold gate it reported, and
`../evaluation/RESULTS.md` measures why that gate went - it admitted **2%** of
unconditional designs where **41%** were designable. Budget about a minute per design, and
since every target asks for 100, score a subset (`--tracks`, or a trimmed split) often.

## What this is, and what it is not

Upstream, unmodified, with two additions and one subtraction.

**Added:** the bridge between this task's request format and upstream's, because
upstream trains from structures and designs from its own `InputSpecification`, neither of
which is this task's `.npz`:

| file | what |
| --- | --- |
| `rfd3_inputs.py` | request `.npz` -> upstream `InputSpecification` + a conditioning structure rebuilt from the context |
| `rfd3_collect.py` | output mmCIF -> one PDB per design plus `<track>.jsonl`, recovering `motif_map` for enzyme by finding where the model actually put the motif, which is the honest reading of an unindexed design |
| `rfd3_infer.sh` | the three stages end to end, with `rfd3 design` untouched in the middle |
| `infer.sh` | the contract entry point; finds a checkpoint and calls `rfd3_infer.sh` |

**Subtracted:** widths, via `configs/experiment/pretrain_small.yaml`. Block counts,
condition mix, crop rules, transforms, losses, optimizer, schedule, EMA and batching are
all upstream's, untouched.

| | upstream `pretrain` | `pretrain_small` |
| --- | ---: | ---: |
| `c_s` / `c_z` / `c_atom` | 384 / 128 / 128 | 128 / 32 / 64 |
| `c_atompair` | 16 | **16, unchanged** |
| `c_token` / `c_t_embed` | 768 / 256 | 192 / 128 |
| upcast / downcast `c_model` | 128 | 64 |
| token transformer blocks | 18 | **18, unchanged** |
| atom encoder / decoder blocks | 3 / 3 | **unchanged** |
| pairformer blocks | 2 + 2 | **unchanged** |
| recycles | 2 | **unchanged** |
| **parameters** | 168,038,994 | **12,445,554** |

Two of those are load-bearing and should not be "tidied":

- **`c_atompair` stays at 16.** It carries the reference-conformer geometry into the atom
  pair representation. Narrowing it is what cost an earlier hand-written port its bond
  geometry.
- **`c_token` must be a multiple of 48.** The upcast splits it into `n_split: 3` chunks
  and the diffusion transformer runs 16 heads over the result; upstream's 768 is 48x16,
  and 192 is 48x4. 256 fails inside einops.

**Architecture and recipe only - no RFdiffusion3 weights are used.** Every parameter is
randomly initialised and trained in-session, as the contract requires.

## Training

**One config, `experiment=pretrain_small_ppi`.** Both the agent and the operator run it;
they differ only in scale and in where the data lives.

| | agent | operator |
| --- | --- | --- |
| budget | 1 GPU, 24 h | 8x L40S, 48 h |
| data | staged into the image, `/data/fit` (the config's default) | the full PDB corpus, passed as `paths.data.*` overrides |
| how | `train.py experiment=pretrain_small_ppi` | `submit_rfd3_small_ppi.sh` |

The config defaults to the container, so the agent runs it with no overrides at all. The
operator's script supplies the cluster's mirror and split directory plus 8 ranks. Nothing
about the model or the recipe differs between them - against upstream `pretrain` the
resolved model differs in exactly the **7 width leaves**, and the only other change is
`ppi.frequency: 0.0 -> 5.0` (both verified by composing and diffing).

**`submit_rfd3_small_ppi.sh` is the exact recipe behind the numbers in
`../evaluation/RESULTS.md`** - it is kept here so the reported row is reproducible rather
than anecdotal. Its site paths are `${FOUNDRY}`, `${WORK}`, `${PQ}` and `${VENV_PY}`;
everything else is portable. It documents the three things that are easy to get wrong:
one rank per GPU **launched by srun** (`trainer.devices_per_node=8` alone silently trains
on 1 of 8), a short `TMPDIR` (dataloader workers bind an AF_UNIX socket and the path has
a 108-character limit), and `calculate_hbonds=0` (upstream's 0.2 raises without HBPLUS).

`pretrain_small_ppi` differs from `pretrain_small` by **one leaf**,
`ppi.frequency: 0.0 -> 5.0`. Upstream's pretraining stage sets it to zero - PPI is meant
to come from a later fine-tuning stage - which is why every ppi_binder evaluation before
this change read exactly 0.000. Frequencies normalise per example over the conditions
valid for it, and PPI requires a binary protein-protein interface, so the override
changes the mix only on those examples: there, tipatom and ppi get 34% each, against
tipatom's 53% and ppi's 0% before.

**Read `TRAINING_DATA.md` before running either.** The one thing that could not be
vendored is the data contract: what the two index parquets must contain, what `cluster`
means, and the offline gotchas that upstream's own README does not cover (its pointer to
the AtomWorks dataset documentation is a URL, and the runtime has no network - that page
is vendored at `../references/atomworks-datasets.md`). Its **§0** covers the other case:
with network, upstream's real metadata tables are a public download
(`atomworks setup metadata`), which is faster than rebuilding them and keeps upstream's
own clusters.

### What the recipe spends its steps on

Worth knowing before changing it, because the defaults look surprising.

**`tipatom` is over half of pretraining** - frequency 5.0 against 2.0 for island - and it
is not a variant of island. Islands of length **one**, 2 to 12 of them, and within each
residue only a *bonded subgraph* is shown: seeded at the atom furthest in bonds from the
backbone O (the functional tip) with probability 0.8, then expanded by
`Geometric(1/(1+3)) - 1` bonds. The model is handed a few scattered sidechain tip atoms
and has to build a protein around them, which is this benchmark's enzyme track.

**`p_unindex_motif_tokens: 0.5`** on both island and tipatom puts the motif on its own
chain with no position in the designed chain, so the model chooses the placement. This
matters here because the benchmark's motif requests do not specify positions either, and
it is why `rfd3_collect.py` recovers `motif_map` from the output rather than deciding it
up front.

**One optimizer step is 48 noise samples** (16 diffusion draws x 3 accumulation passes)
from 3 structures at a 256-residue crop. RFD3 spends its steps on many noise draws of
few structures, so a 1,000-step warmup is a long time in wall clock and the loss is
noisy per batch by design.

**Conditioning is part of the model, not a post-process.** Motif, ligand and target atoms
sit in the *same* atom array as the design, held at noise level zero via
`t_L = t * (~is_motif_atom_with_fixed_coord)`, so a fixed atom passes through untouched
while the network still attends to it. A motif is scaffolded *by the model*; nothing is
grafted, docked or seated afterwards.

### Reading the loss

`mean_lddt_*` is a **loss** (`1 - lddt`, `metrics/losses.py:299`) - lower is better, and
`mean_lddt_dna: 1.0000` just means the batch had no DNA. `mse_loss_mean` near **1.0** is
the score of a network that predicts nothing, at every noise level, because the objective
is EDM-preconditioned; a run sitting there has learned nothing however smooth the curve
looks.

`mean_lddt_protein` falls early and then sits flat for a long time. That is what the
metric does here, not a regression. Judge progress on designability, or on
`insertion_rmsd` from a design's own `.json` - a cheap continuous capacity signal with a
known target of ~0.7 A from upstream's released 168M model.

## Inference

`infer.sh INPUT_ROOT OUTPUT_ROOT` is the contract entry point. The contract gives it no
slot for a checkpoint, so the location is a convention: `$RFD3_CKPT`, else
`ckpt/*.ckpt` beside the script, else `*.ckpt` beside the script, newest first. It stages
through a scratch directory and lifts `predictions/` to `OUTPUT_ROOT`, which is the
layout `contract.md` specifies.

**Train into the package.** The verifier runs `infer.sh` from an unspecified working
directory and may relocate the package first, so nothing outside it is reachable:

```bash
cp -r /workspace/baseline/rfd3-small/. /workspace/submission/
```

Optimizer state must stay out of the package - `instruction.md` requires the submission
to carry one current candidate and no optimizer state - so keep training output under
`/workspace/research/` and copy or symlink only the chosen checkpoint into
`submission/ckpt/`.

**You must write `harness.json` yourself.** Nothing in this directory emits it, and
`main_verifier.py` requires it alongside `infer.sh` and `README.md`. Declare
`kind: trained_generator`, the checkpoint as a learned component with its sha256, and its
parameter count - which for an unmodified `pretrain_small` is 12,445,554, and which you
can recount from a checkpoint without loading the model, since each one carries its own
resolved `train_cfg`.

**Your package must resolve its own imports.** The verifier launches `infer.sh` through
`subprocess.Popen(env=...)`, which **replaces** the environment rather than extending it,
so the image's `PYTHONPATH` and `PROJECT_ROOT` are *not* present while your script runs -
and the image reaches vendored `rfd3`/`foundry` through `PYTHONPATH` alone, because it
installs their dependencies by name and deliberately does not `pip install rc-foundry`.
`infer.sh` therefore exports both itself, derived from its own location:

```bash
export PROJECT_ROOT="$here/rfd3"
export PYTHONPATH="$here/rfd3/src:$here/rfd3/models/rfd3/src${PYTHONPATH:+:$PYTHONPATH}"
```

Package-owned paths go first, so a relocated copy uses its own vendored source. Keep this
if you restructure the package, and add any directory of your own the same way. A
submission that imports fine in an interactive shell and dies under the verifier is
almost always this.

## Layout

| File | What it is |
| --- | --- |
| `rfd3/` | upstream RFdiffusion3 and RoseTTAFold3 source, pinned at `b02eed6a`. The model, the recipe, the transforms, the configs |
| `rfd3_inputs.py` | request `.npz` -> upstream `InputSpecification` |
| `rfd3_collect.py` | upstream output -> the task's submission format |
| `rfd3_infer.sh` | the three inference stages, upstream untouched in the middle |
| `infer.sh` | contract entry point; checkpoint discovery + `rfd3_infer.sh` |
| `submit_rfd3_small_ppi.sh` | the 8x L40S operator recipe behind `RESULTS.md` |
| `TRAINING_DATA.md` | the data contract: schema, clusters, and how to get the corpus |
| `residues.py` | the atom14 table, byte-identical to the grader's copy |

`residues.py` is pinned: `tests/test_task.py` asserts it matches `tests/residues.py`
byte for byte, so the submission and the grader can never disagree about atom ordering.

## Rules that still apply

- One system serves all four tracks. Upstream already does this - motif, ligand, target
  and hotspot atoms all enter as atoms carrying their chemistry and their role, with no
  track-specific branch.
- `harness.json` declares the paradigm. Add a pretrained predictor to the loop and it
  becomes `hybrid`; replace training entirely and it becomes `predictor_inversion`.
- Declare pretrained assets rather than shipping them; the operator mounts them at
  `/pretrained`.
- **Benchmark the model, not a repair pass.** Score the sampler's own output. Anything
  that keeps only a CA trace and rebuilds a chain around it reports the quality of that
  projection, not of what the network learned. The CA-CA gate that used to reward exactly
  that move is gone, which removes the incentive without changing the principle.
