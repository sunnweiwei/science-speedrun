# Reference design systems

This task starts from real design systems, not an empty file. Read the executable
starter first, then use these references to decide what to reproduce, retain, or
replace.

## Pinned upstream source trees

The development image contains writable, network-independent source snapshots:

| Path | System | Paradigm | Pinned revision | License |
| --- | --- | --- | --- | --- |
| `/opt/references/rfd3` | RFdiffusion3 | trained generator | `b02eed6a6bdf8f44d14a80cc36e3da13c9f2291c` | BSD-3-Clause |
| `/opt/references/boltzgen` | BoltzGen | trained generator | `a3149cf18eeb58648d1abbb27539bd73f746cdda` | MIT |
| `/opt/references/disco` | DISCO | trained generator (co-design) | `82b594f838eb61dd8c78ae3a403ccb8f00cf7abf` | Apache-2.0 |
| `/opt/references/proteina_complexa` | Proteina-Complexa | flow matching + test-time search | `33b76585801eb902802cc7410c85bc37ae81b346` | see below |
| `/opt/references/boltzdesign1` | BoltzDesign1 | predictor inversion (trains nothing) | `31aeedd7f187ba88dd9e92514070aee13381cb93` | MIT, asserted in README only |

Proteina-Complexa is a multi-license repository: **code Apache-2.0, datasets CC BY 4.0,
weights under the NVIDIA Open Model License**, which states that models are commercially
usable and that derivative models may be created and distributed. Its `licenses/`
directory carries the full terms and the third-party notices (openfold,
alphafold3-pytorch, ProteinMPNN, LigandMPNN, ColabDesign). Read them before shipping
anything derived from it. Note that the *original* Proteina repository is research /
non-commercial only; Complexa is not, and the two should not be confused.

**BoltzDesign1 is vendored as of 2026-09-23** (it previously was not). It trains nothing
at all: it inverts a pretrained Boltz-1 at inference, optimising the distogram through
the Pairformer and confidence modules. It is the reference for the
`predictor_inversion` paradigm, and the reason this task grades a harness rather than a
trained model. **Licensing caveat:** its README states "MIT License - See LICENSE file
for details", but the repository ships **no LICENSE file**, so the README is the only
license grant on record. GitHub's own licence detection reports none. Note also that it
bundles its own copies of `boltz` and `LigandMPNN` subtrees.

All five systems are source-only and **no generator weights are supplied**, for any of
them. BoltzDesign1 in particular is inert without pretrained Boltz-1 weights, which this
task does not ship - it is here to be read, not run end to end.

Machine-readable specifications for all five systems are in `model_specs/`. Published criteria
and results are in [../evaluation/RESULTS.md](../evaluation/RESULTS.md), which also
carries the criteria this task grades on and the row the starter measures.

## Where to start

`/workspace/baseline/rfd3-small` is upstream RFdiffusion3 reduced in layer widths, runnable and
trained from scratch in-session. Designability is the only thing it is scored on - there
is no geometry or conditioning gate to clear - and a real checkpoint of it lands non-zero
but low on every track: enzyme 0.054, sm_binder 0.030, ppi_binder 0.004, unconditional
0.410. See [../evaluation/RESULTS.md](../evaluation/RESULTS.md). It exists to establish
the interface and a measured floor, not to be competitive.

The five vendored systems span genuinely different paradigms, and the
choice between them is part of the research:

- **All-atom diffusion** (RFdiffusion3) - denoise a 14-atom-per-residue representation
  with atom-level conditioning.
- **Generative binder framework** (BoltzGen) - all-atom generation against arbitrary
  biomolecular targets.
- **Sequence-structure co-design** (DISCO) - generate sequence and structure jointly
  with cross-modal recycling, rather than backbone-then-inverse-folding.
- **Predictor inversion** (BoltzDesign1) - train nothing; optimise inputs through a
  frozen pretrained structure predictor at inference.
- **Flow matching with test-time search** (Proteina-Complexa) - partially-latent flow
  matching over a fully atomistic representation, with inference-time optimisation.

This task's contract accepts all of them. `harness.json` declares which one you built.

## Scientific boundary

Reference **source and configurations may be copied or changed**. What is supplied here
is code and architecture, not a solution.

Reference **learned weights are not supplied**. Vendoring a finished design system's
checkpoints would reduce this task to selecting among them, which measures nothing.
Pretrained models are nonetheless permitted in a submission - a `predictor_inversion`
system is nothing but a pretrained predictor - and must be declared in `harness.json`
with a digest, a license, and a role. The operator mounts declared assets at
`/pretrained`.

The distinction that matters: **components are supplied, finished generators are not.**
A pretrained structure predictor is an ingredient; a trained backbone generator is the
deliverable.

## Evaluation instruments are not references

LigandMPNN at `/opt/ligandmpnn` and RoseTTAFold3 at `/workspace/evaluation/rf3` are the
instruments that score you; their parameters are in `/assets`. Run them to measure your
own designs - `python /workspace/evaluation/score.py --input /data/tune --designs ...`
is the whole workflow, and it needs no network. If your system
depends on them at inference, declare them - but be aware that optimising directly
against the checkpoints that grade you inflates in-silico designability without
necessarily producing better proteins, and the verifier records when a submission
carries them.
