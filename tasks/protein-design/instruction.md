# Develop one all-atom protein design system

## Task and goal

Design, build and keep improving **one** conditional design system that serves all four
tracks in the mounted data: `unconditional` monomer generation, `sm_binder` design around a
small molecule, `ppi_binder` design against a target epitope, and `enzyme` scaffolding of a
catalytic motif.

One system must serve every track. Four independently built pipelines behind a common wrapper
do not satisfy the objective; shared components with track-conditioned behaviour do.

A design counts when it is **designable**: LigandMPNN designs sequences for your structure,
RoseTTAFold3 refolds the complex, and the result must satisfy that track's published
criterion. That criterion is the whole measurement — the task adds nothing to it. The exact
rule differs per track; `/workspace/contract.md` has all of them.

Each track is reported as a **pass rate**, the fraction of requested designs that satisfy it,
with a **diversity** number beside it. There is no single combined score: the tracks apply
different published criteria, so they are compared individually.

This is the whole difficulty. Producing a chain with correct bond lengths that threads a
catalytic motif is mechanical, and the baseline already does it for every design — which is
why none of that is scored. Producing one that a sequence actually folds into is the research
problem. The baseline's measured pass rates are low but not zero on every track, a few percent
on the conditioned tracks, and that is what to beat.

Each target asks for **20 designs** (RFdiffusion3's protocol asks for 100; a pass rate is a per-target
fraction either way), so a pass rate is a per-target fraction out of 20. Designs you do not emit stay in
the denominator.

A passing design must also be **novel**: its maximum TM-score to every chain of the fit corpus you train
on (`/workspace/data/fit/structures`) must be below 0.8, measured with Foldseek (`$FOLDSEEK`, TM-align
mode) as `/workspace/evaluation` describes. Retrieving or lightly deforming training backbones does not
count, however designable they are.

Designs are **all-atom**: up to 14 heavy atoms per residue, backbone in slots 0–3. Motif
residues must carry the requested residue identities and their sidechains are compared, so
catalytic geometry is scored where it actually lives.

The tracks are not symmetric, and the difference matters. `enzyme` hands you a catalytic
constellation to scaffold. **`sm_binder` hands you a bare ligand and nothing else** — no
pocket, no template — and you build a protein around it. `ppi_binder` hands you a target chain
and hotspots to engage. `unconditional` reports diversity beside its pass rate, because with
no conditioning to satisfy, a generator that emits one good fold repeatedly reaches a high
pass rate while having learned almost nothing; the pass rate cannot see that and diversity
can. Rotated or translated copies do not count as distinct — the clustering superimposes
before comparing.

## Workspace

- `/workspace/contract.md`: the package, the invocation, the input and output formats, and
  exactly how designs are scored.
- `/workspace/data/` (read-only; also reachable as `/data`):
  - `README.md` describes how the tracks were built and what each context holds;
  - `fit/` and `tune/`: the development splits, whose native structures are training data.
- `/workspace/baseline/rfd3-small/`: the verified, runnable baseline — upstream RFdiffusion3
  reduced in layer widths — with a README, RESULTS.md and `publish.py`.
- `/workspace/references/`: pinned reference design systems, their published results, and
  where the weights boundary lies. Reference **source** is also under `/opt/references`.
- `/workspace/evaluation/`: `score.py` and the scoring library, for measuring yourself during
  development with the same code the verifier uses.
- `/workspace/submission/`: the current package; empty at start.
- `/workspace/research/`: writable space for experiments, checkpoints and notes.
- `/pretrained/`: where declared pretrained assets appear, at the same path here and in
  verification. Empty unless the operator mounts something.

## Rules

- **One declared system.** The graded artifact is declared in `harness.json` as
  `trained_generator` (you train a generative model, as RFdiffusion3 does),
  `predictor_inversion` (you optimise against a pretrained structure predictor at inference
  and train nothing, as BoltzDesign1 does) or `hybrid`. All three are legitimate. There is no
  parameter limit and no from-scratch requirement. What is enforced is that the manifest
  matches reality: declared learned components exist in the package with the stated digest,
  and declared pretrained assets exist in `/pretrained` with the stated digest, a license and
  a role.
- **Declare pretrained assets rather than shipping them.** Undeclared pretrained weights
  inside the package are a contract violation.
- **Reference weights are not supplied.** Components are given; finished generators are not.
  Reference source may be copied or changed freely.
- **The grading instruments are available to you.** LigandMPNN and RoseTTAFold3 ship in this
  image and you may run them to evaluate your own designs; if your system depends on them at
  inference, declare them. Be aware that optimising directly against the checkpoints that
  score you inflates in-silico designability without necessarily producing better proteins,
  and the verifier records when a submission carries them.
- **The evaluation view contains no native structures**, so there is nothing to look up. Do
  not try to obtain evaluation targets or their identities.
- Do not fit parameters at inference time, and do not make the package depend on the live
  training process, an unrecorded cache or an absolute path.
- There is no network access. Everything must come from the mounted data, this image, or what
  you build into the submission.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission` from
early on, starting with the first trained checkpoint. It holds exactly one current candidate:
no `candidate-vN` directories, optimizer state or superseded weights. The package may be read
at any moment, so replace it atomically:

```bash
python /workspace/baseline/rfd3-small/publish.py DIR --check
```

which stages the copy next to the target, validates `harness.json` the way the verifier does,
and swaps it in with one rename. Keep experiment history under `/workspace/research`.

The verifier runs `infer.sh` once over every evaluation request, and a crash voids every track.
Before each publication, run `python /workspace/evaluation/check_package.py --package DIR`: it
replays the package on every `/data/tune` request and fails unless each one gets a design that
passes the verifier's contract checks (`/workspace/contract.md`, "Failure policy").

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Establish a measured baseline, form a hypothesis, change one thing,
evaluate every affected track, and publish when the evidence improves. Then start the next
experiment. Scoring is self-service and unrestricted; this task has no hidden labels.

**Score the real metric — it is the only one `score.py` offers.** Designability (LigandMPNN
inverse folding plus RoseTTAFold3 refolding) runs in this image with parameters already at
`/assets`, so nothing has to be mounted or downloaded. There is no geometry-only shortcut: one
existed while AlphaFold 3's non-redistributable parameters kept the real scorer out of the
image, and it was deleted once grading stopped gating on geometry, because without that gate
it called every parseable design designable. Budget about a minute per design and score a
subset often (`--tracks`, or a trimmed split); even at 20 designs per target a whole split is
hours. Keep concise notes in `/workspace/research`. Do not stop to hand in a final answer; a
published system or a reached pass rate is a milestone, not the end of the task.
