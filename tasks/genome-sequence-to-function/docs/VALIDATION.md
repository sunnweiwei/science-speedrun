# Validation record (integration of PR #14, 2026-10-01)

Everything below was run on one 8× H100 80 GB host (208 cores, 1.8 TB RAM) from this repository.
The images were built from the committed Dockerfiles. The scores come from the isolated verifier
(`tests/`) unless stated otherwise.

## 1. What the integration changed, and why

PR #14 (v3) was a working pipeline. Several of its choices limited the task:
- 3,200 of the 41,020 eligible training intervals (8 %);
- a 100-GiB data cap and a 50M-parameter cap;
- dense splice outputs of about 600 GB per test run;
- no variant evaluation.

| v3 | problem | v4 |
|---|---|---|
| 3,200 train intervals, float32 targets, 100-GiB cap | far below AlphaGenome's scale; the full set in float32 would be about 1.3 TB | all 41,020 eligible intervals, 8-bit train targets with per-track ranges: 330.7 GB (validation and test unchanged, full precision) |
| 50M-parameter cap | an arbitrary limit unrelated to the budget | no cap; the baseline is sized to train in about 5 h on 8 GPUs (215M parameters) |
| dense `splice_usage` `[196608, 734]` and junction cube | about 636 MB per example, 600 GB per test run | sparse lists (≤ 4,096 sites, ≤ 16,384 pairs); unlisted hidden sites and pairs score as zero; 11.5 MB per example |
| track Pearson only | says little about what the model is for | 13 variant and perturbation evaluations from AlphaGenome and TraitGym with published anchors, reported next to the reward ([EVALUATION_DESIGN.md](EVALUATION_DESIGN.md)) |
| TensorFlow GCS reader | about 6.5 MB/s per process and stalled under concurrency; a full build was infeasible | HTTPS + ISA-L gzip + zero-copy protobuf parsing; byte-identical tensors on all twelve bundles; reconnects mid-stream |
| single-GPU training | could not use the agent's 8 GPUs | DDP (`torchrun -m …train_eval`, `run_ddp.sh`), loopback rendezvous so it works offline |
| grader waited a fixed 600 s for the snapshot | any real package timed out | configurable wait (default 7,200 s) |
| invalid package | no `reward` written | `reward` 0 with `status: invalid_submission` |
| inference limit 42,600 s | longer than the 6 h verifier limit | 4 h for `infer.sh` + `perturb.sh` together, stated in `contract.md` |
| (new in v4) A/B perturbation outputs in float16 | rounds many single-variant effects to zero | float32 in `perturb.py` and the contract |
| no labelled variant data for the agent | the agent could not measure the variant evaluations | development split `variant_evals/valid` (2,697 perturbations, disjoint from test) and `score_variants.py` |

Two further fixes:
- Negative int64 interval starts were decoded as huge unsigned numbers; this is now two's
  complement, with a test.
- The layout now follows TASK_STANDARD: `baseline/`, `references/`, `docs/`, `TASK_*` mounts, and
  no network or GPUs in the agent's compose file.

The author's v3 documents are kept in [v3-author/](v3-author/).

## 2. Data

| step | result |
|---|---|
| train targets, 250 workers | 2 h 52 min, streaming 8.2 TiB of gzip TFRecords at a peak of about 1.4 GB/s; 41,020 examples, 330.7 GB (7.69 MiB per example) |
| export (manifests, labels, 41,020 junction scoring parts) | included in the build; a separate re-export took 8.5 min |
| `validate --splits train,valid,test` | PASS, 19 min; 5,930 tracks; total targets 332.4 GiB |
| variant evaluations | 22 pinned files, SHA-256 verified; 6,587 perturbations |

### Quantization

**Round trip.** On real examples, per-track Pearson in `asinh` space is ≥ 0.9997 (median
0.99999), and the maximum error is half a quantization step.

**Training A/B.** The tiny recipe (`tiny_overfit.yaml`, 3 epochs) was trained on the same 3,200
intervals, once with the v3 float32 targets and once with the new 8-bit targets, on two seeds
each. Validation was scored by `environment/evaluation/score.py` (v4):

| targets | seed 17 | seed 18 | mean |
|---|---:|---:|---:|
| float32 (v3) | 0.1571 | 0.1477 | 0.1524 |
| 8-bit (v4) | 0.1604 | 0.1502 | 0.1553 |

The difference (+0.003) is well inside the seed spread (0.01). The training losses agree to the
third decimal at the same step. Quantization does not cost accuracy at this level.

## 3. Baselines

| system | training | validation (score.py) | test (verifier) |
|---|---|---:|---:|
| tiny reference, author's recipe (0.23M) | 3 epochs on 3,200 intervals, 1 GPU, about 11 min | 0.114–0.160 over five runs | **0.12855** (author's v3 number 0.1424) |
| large reference, 215.3M (the baseline) | 14 epochs on 41,020 intervals, 8 H100, 4 h 9 min | 0.302 | **0.3106** |

On the tiny model, the v3 dense scorer and the v4 sparse scorer give the same test number on the
same checkpoint: 0.12853 and 0.12855.

### Throughput (large reference)

| setting | step (one example per GPU) | examples/s |
|---|---:|---:|
| 1 H100, model only | 0.10 s | 10 |
| 4 H100 DDP, 12 loader workers each | 0.193 s | 20.7 |
| 8 H100 DDP, 12 loader workers each | 0.217 s | 36.9 |

One epoch of 41,020 intervals takes 18.5 min on 8 GPUs.

### Variant evaluations

Per-evaluation numbers for the baseline, on the development split and the test split with their
anchors, are in
[environment/baseline/alphagenome-reference/RESULTS.md](../environment/baseline/alphagenome-reference/RESULTS.md).
Apart from sQTL (0.73) and enhancer–gene linking (0.39, chance 0.20), the baseline is near
chance, far below AlphaGenome's anchors. The evaluation path was checked to rule out an artifact:
- **Variant placement.** For every SNV query, the reference base matches one of the two alleles.
  The large model's A − B difference peaks at the variant's bin, or one 1-kb trunk token away, in
  20 of 24 sampled variants (peak/median 9–142).
- **Output precision.** A/B outputs were first written as float16, which rounds many
  single-variant effects to zero. `perturb.py` now writes float32, and the contract asks for it.
  Rescoring with float32 changed every development metric by at most 0.016, so precision was not
  the cause either.

## 4. Verifier

| run | result |
|---|---|
| tiny package, full verifier | reward 0.12855, all 13 variant evaluations reported (float16 perturbation outputs), 43 min |
| large package (2.5 GB), full verifier | reward **0.3106**, 13 variant evaluations; `infer.sh` 10.1 min + `perturb.sh` 9.7 min of the 4 h budget; 31 min in total |
| broken package (`infer.sh` exits 3) | `reward` 0, `status: invalid_submission`, 4 s |

The unit tests pass:
- 37 source tests in the materializer image (2 skipped);
- 10 baseline tests in the development image.

The development image builds. Its compose healthcheck passes with only the development entries
mounted (no test targets, labels or variant labels). Training and `perturb.sh` run against the
mounted `/workspace/data`.
