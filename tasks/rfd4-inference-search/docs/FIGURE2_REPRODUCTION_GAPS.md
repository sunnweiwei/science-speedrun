# RFD4-Proteina Figure 2 reproduction gap report

**3 of 16 official baseline points have independent local measurements. The remaining 13 include all four Search points.** This inventory covers Figure 2 and supplementary Table S29 of *Biomolecular design through latent reasoning with RFD4-Proteina*. It uses the evidence recorded at commit `c95ee9dd` on November 21, 2026. Each point is one suite evaluated at one of the authors' four inference settings. [Paper](https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1) · [Machine-readable coverage](../environment/references/overall/measured/coverage.json)

## What can already be used as a local baseline?

The DNA suite has three independently measured settings on RTX A6000. Each covers all three official targets and 600 original attempts. The following rates and timings come from our runs. They are the available baseline for comparing a future submission under the same conditions. [Frozen target-level counts and times](../environment/references/overall/reproduced_targets.csv) · [Measured coordinates](../environment/references/overall/measured/points.csv)

| DNA setting | Passing clusters / attempts | Local clustered pass rate | Generation attempts/GPU-h | Attempts/GPU-h including final AF3 |
| --- | ---: | ---: | ---: | ---: |
| Low | 15 / 600 | 2.50% | 673.1 | 247.1 |
| Medium | 12 / 600 | 2.00% | 460.5 | 211.6 |
| High | 17 / 600 | 2.83% | 283.0 | 164.3 |

The archive replay recovers **16/16 author-reported points**. That verifies the arithmetic and plotting of the published results. The S15 sampler runs and the native monomer beam runs establish baselines for a separate, unconditional monomer task. [Archive audit](../environment/references/overall/README.md) · [Monomer results](../README.md#locally-measured-monomer-results)

## Which independent measurements are missing?

Every row below is **open**. The last two columns give the authors' values solely for checking a completed measurement. Leave the local coordinates empty until evidence is available. Generation throughput includes internal search time and excludes the common final AF3 evaluation. [Full-precision author reference, including end-to-end rates](../environment/references/overall/operating_points.csv)

| Suite | Missing setting | Targets | Original attempts | Author clustered pass rate | Author generation attempts/GPU-h |
| --- | --- | ---: | ---: | ---: | ---: |
| Protein binding | Low | 24 | 4,800 | 7.73% | 2,388.3 |
| Protein binding | Medium | 24 | 4,800 | 9.71% | 1,619.0 |
| Protein binding | High | 24 | 4,800 | 10.63% | 985.6 |
| Protein binding | Search | 24 | 4,800 | 14.92% | 57.4 |
| Small-molecule binding | Low | 77 | 15,400 | 12.12% | 7,740.8 |
| Small-molecule binding | Medium | 77 | 15,400 | 12.58% | 5,973.7 |
| Small-molecule binding | High | 77 | 15,400 | 14.32% | 5,242.3 |
| Small-molecule binding | Search | 77 | 15,400 | 20.94% | 110.6 |
| Enzyme scaffolding | Low | 37 | 7,400 | 9.78% | 5,160.8 |
| Enzyme scaffolding | Medium | 37 | 7,400 | 10.61% | 3,745.1 |
| Enzyme scaffolding | High | 37 | 7,400 | 12.07% | 3,495.6 |
| Enzyme scaffolding | Search | 37 | 7,400 | 16.64% | 84.9 |
| DNA binding | Search | 3 | 600 | 4.83% | 111.5 |

At the paper's 200 original attempts per target, the open rows account for **111,000 attempts across 555 target-by-setting records** for one complete measurement of each point. Nine missing one-shot settings account for 82,800 attempts. The four Search settings account for 28,200. The three existing DNA settings supply the other 1,800 attempts of the 112,800-attempt reference field. These totals describe coverage; runtime varies substantially by setting. Each complete row can be verified separately. DNA–Search is the one missing row in the DNA panel.

## When to close a row

1. **Establish provenance.** Retain the independent run ID, model and evaluation revisions, seeds, device, numerical precision, batching and timing convention. Link the frozen measurements to their run logs. Check for an existing independent run before allocating more compute. The current local comparison profile is `rfd4-a6000-warm-v1`. Keep measurements made on different hardware or with cold timings separately labeled.
2. **Verify coverage and counts.** Require the official target roster for the stated suite and setting, with 200 original attempts per target. Keep failures in the denominator. For each target retain the numbers scored, passed and in distinct passing clusters, together with generation GPU-seconds and common final AF3 GPU-seconds. A partial roster remains a pilot. A complete cohort with zero passing clusters is a measured zero.
3. **Recompute the point.** For each target, let `A_i` be attempts, `C_i` distinct passing clusters, `G_i` generation GPU-seconds including internal search, and `F_i` final AF3 GPU-seconds. The clustered pass percentage is the mean over targets of `100 × C_i / A_i`. Generation throughput is `3600 × sum(A_i) / sum(G_i)`. For end-to-end throughput, use `sum(G_i + F_i)` as the denominator. End-to-end unique yield is the mean of `3600 × C_i / (G_i + F_i)` across targets. Record the local rate, its difference from the paper in percentage points, and the throughput ratio with the hardware and timing basis. Report observed variation when independent repeats exist.
4. **Record the outcome.** Mark the row *independently measured* when the evidence and coverage checks pass. Agreement with the paper is a separate conclusion. A different outcome remains a useful local baseline when the discrepancy and protocol are documented.

## How to verify the handoff and update the plots

Deliver a frozen target-level CSV with the same header as [the existing local baseline table](../environment/references/overall/reproduced_targets.csv), along with its provenance and underlying run records. Use a complete set of target rows for each additional setting. The reporting tool accepts several completed settings in one file and rejects conflicts with the pinned measurements. From the task root:

```bash
python environment/references/overall/measured_report.py \
  --baseline-targets /path/to/additional-independent-baselines.csv \
  --output /path/to/new-measured-report
```

Check the new directory's `points.csv` and `coverage.json` against the run records. `generation.png` and `including-af3.png` show the two throughput views. For the current three-point report, omit `--baseline-targets`. [Current measured plot](../environment/references/overall/measured/generation.png) · [Reporter and measurement profile](../environment/references/overall/MEASURED.md)

**Completion for the baseline campaign** means 16 independently supported points, four in each panel. A new Agent submission is a separate measurement: at one operating budget it contributes one candidate point per independently evaluated suite. The baseline figures can be completed before testing a new algorithm. The current live verifier executes monomer requests and can plot sealed official-suite measurements for a matching submission. An automatic end-to-end evaluator for new submissions on all four functional suites remains outstanding. [Submission and reporting contract](../environment/contract.md#independent-official-suite-report)
