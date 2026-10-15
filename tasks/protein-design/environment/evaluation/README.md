# Development evaluation

Scoring is self-service and unrestricted. This task has **no hidden labels**:
designability is an intrinsic property of a backbone, measured by refolding, so
there is nothing to leak by scoring yourself. The verifier's isolation exists to stop
an untrusted model tampering with the scorer, not to hide answers from you.

```bash
# the same pipeline the verifier runs - and the only scoring mode there is
python /workspace/evaluation/score.py --input /data/tune --designs /tmp/designs

# one track, or a trimmed split, while iterating: every target asks for 20 designs
python /workspace/evaluation/score.py --input /data/tune --designs /tmp/designs \
    --tracks enzyme
```

`lib/` contains the exact `common.py`, `geometry.py`, `pipeline.py`, `grader.py`, and
`residues.py` the verifier uses, copied in at image build time, so a development score
and a verifier score differ only by the input split and the machine.

## What lives here

Everything the *scoring* side needs, in one place:

| | |
| --- | --- |
| `score.py`, `lib/` | the scorer and the grader modules the verifier shares |
| `ligandmpnn/` | inverse folding - the ORIGINAL LigandMPNN repo @ `26ec57ac` (MIT), vendored. Both images COPY it to `/opt/ligandmpnn` |
| `rf3/` | refolding - RoseTTAFold3 from foundry @ `b02eed6a` (BSD-3, IPD/UW), vendored. On `PYTHONPATH` in both images |
| `benchmark/` | the **operator** benchmark. NOT in either image - see below |

Two couplings worth knowing before you move anything:

- **`rf3` imports `foundry`**, at 29 sites, and `foundry` lives in the *starter's*
  vendored tree (`environment/baseline/rfd3-small/rfd3/src`). Both must be on `PYTHONPATH` or
  `import rf3` fails at module scope, which breaks every scorer call on every track.
- **The dependency runs both ways.** `foundry.callbacks.train_logging` imports
  `rf3.utils.loss`, so the starter's *training* path needs `rf3` importable too. That is
  why the agent image puts `evaluation/rf3/src` on `PYTHONPATH` alongside the starter's.

**`benchmark/` is excluded from both images** by `.dockerignore`, deliberately, even
though the rest of this directory ships. `targets-full.json` enumerates every
**evaluation** target id — the 50 benchmark targets (41 enzyme + 4 sm_binder + 5
ppi_binder) plus the synthetic `uncond_L100`, i.e. the whole **51-target** evaluation
view that `ingest_targets.py` also builds — and `targets.json` enumerates the frozen
14-target subset. `uncond_L100` is graded but is not part of the RFdiffusion3 benchmark
and has no published criterion, which is why the 50 and the 51 are both quoted around
the task depending on which set is meant. The task is built so an agent
sees only `/data/fit` and `/data/tune`. It also drives the verifier-only evaluation view through
`environment/data/targets/`, which is not in the image either - so shipping it would
leak the target list *and* ship scripts that could not run. Nothing in `benchmark/` is
part of your workflow; it is how an operator produces the reference row in
`RESULTS.md` in this directory.

**The full path works in this image.** Both scoring models are installed and their
parameters are baked in at `/assets`: LigandMPNN (the original repository, pinned) for
inverse folding, and RoseTTAFold3 for refolding. Nothing has to be mounted, and nothing
has to be downloaded at run time. If `/assets` is ever missing, rebuild it anywhere with

```bash
bash environment/fetch_designability_weights.sh /assets
```

This is a change from earlier revisions of the task, which used AlphaFold 3. AF3's
parameters cannot be redistributed, so the real metric could not run in this image and a
`--geometry-only` upper bound was effectively forced. That flag is **gone** as of
2026-09-26, deleted together with the pre-fold gate it reported: without the gate it
called every parseable design designable, which is a wrong number rather than a cheap
one. There is one scoring mode, and it is the one you are graded on.

## What counts as a pass

**The published criterion, and nothing besides it.** Your design is inverse-folded,
refolded, and judged on its track's rule from `success.txt`. There is no pre-fold geometry
or conditioning gate. Until 2026-09-26 there was, and it scored good backbones a flat zero
- it admitted 2% of unconditional designs where 41% were designable - so it was deleted
rather than retuned.

Two things do run before folding, and neither is a criterion this task invented:

- **enzyme's ligand-clash term** - `success.txt` criterion 2: no non-motif backbone atom
  (N, CA or C) within 1.5 A of the closest ligand atom. The published criterion measures
  it on the raw design, so that is where it is measured. Checking it first only saves a
  fold, since it is a conjunct of the verdict either way; on the reference run it rejected
  52% of enzyme designs.
- **contract validation** - one chain, exactly the requested `design_length`, atom names
  valid for their residue type, finite coordinates, and on enzyme a `motif_map` naming one
  distinct in-range residue per motif position carrying the requested identity. Without
  that last part motif recapitulation has nothing to measure against. This is
  `contract.md`, not scoring.

Nothing judges your sidechain rotamers any more. The protocol *replaces* your sequence
by inverse folding, so a backbone-only submission is valid and always was -
`parse_allatom_pdb` builds its mask from whatever atoms are present.

### Designability

Inverse-fold the backbone, refold the sequences, compare. **4 sequences per backbone on
every track** - upstream uses 8 for enzyme and sm_binder, so this is a documented
deviation, and since a design passes if ANY sequence passes it can only lower a pass rate.
The design passes if any sequence (and, for enzyme, any of its 5 predictor models)
satisfies the track criterion.

| track | inverse folding | models/seq | passes when |
| --- | --- | ---: | --- |
| unconditional | ProteinMPNN | 1 | backbone RMSD <= 1.5 A |
| sm_binder | LigandMPNN | 1 | backbone RMSD <= 1.5 A AND ligand RMSD <= 5 A AND interface PAE <= 1.5 AND ipTM >= 0.8 |
| ppi_binder | ProteinMPNN | 1 | binder backbone RMSD <= 2.0 A AND interface min PAE <= 1.5 |
| enzyme | LigandMPNN | 5 | motif all-atom RMSD <= 1.5 A after motif-backbone alignment |

LigandMPNN is used exactly where there is a ligand to condition on; ProteinMPNN
elsewhere. Sampling temperature 0.1.

### Per-track numbers

`pass_rate` is the mean over targets of (passing designs / requested designs) - a
per-target mean, so a target with many designs does not dominate one with few. Every view
requests **20 designs per target** (RFdiffusion3's published protocol requests 100), so
each target's rate is out of 20. Every track is scored the same way; there is no
per-track weighting and no blend.

`diversity` is reported next to it on every track: the mean over targets of (distinct
structural clusters among that target's passing designs / its passing designs), clustered
at a 2.0 A RMSD radius. It matters most on `unconditional`, where there is no conditioning
to satisfy and a generator emitting one designable fold over and over reaches a pass rate
of 1.0 - the pass rate cannot see that, and diversity can.

**There is no composite score.** The four tracks apply different published criteria and
are not commensurable, and `unconditional` is synthetic to this task rather than part of
RFdiffusion3's benchmark, so averaging them would produce a number this task invented
rather than one it measured - dominated, moreover, by the one track nobody publishes.
`reward.json` carries the per-track pass rates, diversity, and design counts.

This also removed a real defect: while a composite existed, it was an equal-weight mean
over four tracks while the evaluation view supplied targets for only three, so an empty
`unconditional` track scored 0.0 and capped the headline at 0.75. Nothing is capped now,
because nothing is aggregated - and since 2026-09-28 the view populates that track too
(`uncond_L100`, synthesised at length 100), so the count no longer disagrees either.

The per-track spec sheets are `environment/data/targets/{enzyme,sm,ppi}/success.txt`,
and every threshold above lives in one place: `lib/common.py`.

One caveat to record when you report numbers: the per-track thresholds come from
RFdiffusion3's published `success.txt`, which was calibrated with AlphaFold. Scored with
RoseTTAFold3 they do not carry that published meaning, so these numbers are internally
comparable but not directly comparable with the RFdiffusion3 paper.

Folding is the slow part - budget roughly a minute per design, since every sequence for
a design is folded in one batched call, and even at 20 designs per target a whole split is
hours. Start with `--tracks enzyme` or a trimmed split while iterating.
