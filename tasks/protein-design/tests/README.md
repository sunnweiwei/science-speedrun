# Independent verification

`test.sh` is Harbor's verifier entry point. In a fresh environment it validates the
evaluation view, the submission, and its `harness.json`, runs `infer.sh` as an
unprivileged user with no network, and seals the design index plus every referenced
structure into a read-only, hash-manifested snapshot. The trusted `grader` service then scores that snapshot and
writes `/logs/verifier/reward.json`.

## Two services, one boundary

The submission generates structures. It never sees LigandMPNN, RoseTTAFold3, or the
scoring code: those live in the `grader` service, which reads only sealed structures
and the public conditioning contexts and never imports the submission.

This task has **no hidden labels** - designability is intrinsic - so the boundary is
not there to hide answers. It is there so an untrusted system cannot tamper with the
instrument that measures it.

Pretrained predictors are permitted, so carrying the grading checkpoints is recorded
rather than rejected: `harness.json` in the verifier output lists any that are present,
because a system optimising against the checkpoints that score it inflates in-silico
designability. See the task README's note on this limitation.

## Scores

`reward.json` carries every per-track metric flattened with underscore-separated names:
`<track>_pass_rate`, `<track>_diversity`, and the design counts behind them.

| Track | `pass_rate` | `diversity` |
| --- | --- | --- |
| all four | mean per-target fraction of requested designs satisfying the published criterion | fraction of passing designs in distinct 2.0 A clusters |

Every target requests **20 designs** (RFdiffusion3's published protocol requests 100; the
per-target pass rate estimates the same quantity), so the evaluation view is 51 targets x 20 =
**1,020 designs**, and grading them folds ~8,600
structures. Grading folds one design per visible GPU (`grader.py --workers`, default
all of them) and `task.toml` asks the verifier for 8, since designs are independent and
each fold is already its own subprocess. Worker count cannot move a score: verdicts are
replayed in submission order, because `cluster_designs` is a greedy walk and `diversity`
would otherwise depend on which fold finished first. The 48 h `[verifier] timeout_sec`
covers the rest, with `main_verifier.py --grader-timeout` and `grader.py --wait-seconds`
at 129,600 s so a correct run is not killed before the verifier's own budget expires. The
rate behind those numbers is 8.9 s/fold measured on an RTX A4000 - ~106 h on one card,
~13 h across 8; the declared H100 has not been calibrated, so the margin is an estimate.

**There is no composite score and no single headline number.** The four tracks apply
different published criteria and are not commensurable, and `unconditional` is synthetic
to this task rather than part of RFdiffusion3's benchmark - so a mean over them would be
invented rather than measured. Compare tracks against tracks.

**`unconditional` is the one track whose target is synthesised, not ingested.**
RFdiffusion3's benchmark has no unconditional target, so a request there is a design
LENGTH and nothing else: `ingest_targets.py::UNCONDITIONAL_TARGETS` emits the single
`uncond_L100` row (length 100, a context holding `design_length` alone), mirroring the
frozen operator definition in
`environment/evaluation/benchmark/standard/targets.json`. Both paths build the same
`{"length": "100-100"}` spec, so the graded number is comparable with RESULTS.md's
operator row on this track as well as the other three.

It was an empty file until 2026-09-28, on the reasoning that a synthesised length does
not belong in a graded view. That was withdrawn because it contradicted the task as
delivered: `instruction.md` tells the agent to build a system serving **all four tracks
in the mounted data**, `contract.md` publishes a passing criterion for the track, and
`materialize.py`'s practice view has always populated it - so an agent developed against
four tracks and was graded on three. Populating the graded view costs ~400 of ~43,000
folds, under 1%. **`unconditional` is still synthetic to this task** and still has no
published upstream number, which is why it is compared track-against-track and never
averaged in; that argument never depended on the track being empty.

Diversity is reported beside the pass rate, not blended into it. Mode collapse does not
show up in a pass rate, since one designable fold emitted repeatedly passes every time;
on `unconditional` diversity is what separates a generator from a lookup table.

A design passes when it is **designable**: LigandMPNN (or ProteinMPNN) sequences refolded
by RoseTTAFold3 satisfy the track's published criterion, which for enzyme includes motif
sidechain recapitulation on the refolded model. The grader adds nothing to that criterion.
Two things run before folding and neither is an addition: `success.txt`'s enzyme
ligand-clash term, which the published criterion measures on the raw design anyway, and
contract validation of the request - the `motif_map` must name one distinct in-range
residue per motif position carrying the requested identity, or recapitulation has nothing
to measure against. The geometry and conditioning gate this section used to describe was
deleted on 2026-09-26; `environment/evaluation/RESULTS.md` tabulates what it cost.

Designs that were never emitted stay in the denominator. A scorer that cannot run
exits nonzero and writes no `reward.json` at all, so infrastructure failure is never
reported as a measured zero.

## Local interface checks

```bash
bash protein-design/tests/run_unit_tests.sh
```

23 tests over synthetic all-atom structures and a stubbed designability scorer. They
cover the input contract, tamper detection, request validation, motif
residue-identity enforcement, the published enzyme ligand-clash term, the harness manifest
across all three paradigms, the failure policy, path escapes, mode collapse, and the
trivial-generator check. The conditioning criteria are asserted through the stubbed
scorer, since that is where they now live. They need only NumPy.

They deliberately do **not** load LigandMPNN or RoseTTAFold3 weights, so they establish
no benchmark result. A real score needs the evaluation view, the fetched `/assets`, and a
trained submission.

They also call `grader.score_track` directly rather than `grader.main()`, so the entry
point that writes `reward.json` is not covered by them. Exercise it explicitly when
changing the grader's output.

## Running the grader outside the image

The defaults (`/assets`, `/opt/ligandmpnn`) are what both Dockerfiles set, so inside a
container nothing needs passing. Reproducing a known answer on another cluster does:

```bash
export PYTHONPATH=<task>/environment/baseline/rfd3-small/rfd3/src:<task>/environment/evaluation/rf3/src
python tests/grader.py --sealed SEALED --contexts VIEW --output OUT \
    --assets /path/to/fetched/assets \
    --ligandmpnn <task>/environment/evaluation/ligandmpnn
```

`--assets` and `--ligandmpnn` also read `PROTEIN_DESIGN_ASSETS` and `LIGANDMPNN_ROOT`,
matching `environment/evaluation/score.py`. Do not run it from `/tmp`: a phantom
`matplotlib` directory there makes torchmetrics believe matplotlib is installed and then
fail on `matplotlib.axes`.

**Off-image scoring needs one patch the image applies and `--ligandmpnn` does not.**
LigandMPNN vendors a copy of openfold predating NumPy 2 which still uses `np.int`,
`np.object` and `np.bool`, and `run.py` imports it at module scope — so pointing
`--ligandmpnn` at a bare checkout dies at import on *every* track with
`AttributeError: module 'numpy' has no attribute 'int'`. The fix lives only in
`environment/Dockerfile`, deliberately, so `environment/evaluation/ligandmpnn` stays
diffable against upstream. Apply it to a **copy**, not to the checkout:

```bash
cp -r <task>/environment/evaluation/ligandmpnn /somewhere/ligandmpnn-patched
sed -i 's/\bnp\.int\b/int/g; s/\bnp\.object\b/object/g; s/\bnp\.bool\b/bool/g' \
    /somewhere/ligandmpnn-patched/openfold/np/{residue_constants.py,relax/utils.py} \
    /somewhere/ligandmpnn-patched/openfold/data/templates.py
```

`fetch_designability_weights.sh` covers the weights but not this, so it is not by itself
the whole portability story.

### The known answer to reproduce

A full graded pass on the reference checkpoint, at the real protocol of 51 targets × 100
designs. A build that lands here is scoring the same quantity as the operator harness; one
that does not is misconfigured rather than unlucky.

| track | `pass_rate` | `diversity` |
| --- | ---: | ---: |
| enzyme | 0.0624 | 0.341 |
| sm_binder | 0.0275 | 0.750 |
| ppi_binder | 0.002 | 0.200 |
| unconditional | 0.420 | 1.000 |

`conditioning_satisfied` is 100% on all four tracks, and the pass takes **9 h 33 min on
8× L40S**. The checkpoint, the packaged submission and both JSON artifacts are outside the
repo at `/net/tukwila/yanjing/verifier-ref-fixed/`; ask before assuming a local path. The
operator-side comparison and the five spec bugs that had to be fixed before the two paths
agreed are in `environment/evaluation/RESULTS.md`.

Packaging a submission to replay this also means writing `harness.json` by hand; nothing
in the starter emits it. See `environment/baseline/rfd3-small/README.md`, "You must write
`harness.json` yourself".
