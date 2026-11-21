# RFD4-Proteina inference search

## Overview

The primary scientific comparison uses the official four-suite RFD4 evaluation in Figure 2 and supplementary Table S29. The offline scorer reconstructs all 16 published Low, Medium, High and Search operating points from 564 target records. It also accepts independent, controller-owned per-attempt or per-target measurements and plots their quality–throughput frontiers. [Official metrics, reference curves and starting points](environment/references/overall/README.md)

![Four-suite RFD4 quality–throughput comparison with other published methods](environment/references/overall/quality-throughput-with-comparators.png)

The 16 RFD4 points form the measured frontier among the 42 included protocol/track points under the paper’s conditioning and timing conventions. [Clean rebuild and provenance](environment/references/overall/README.md#rebuild-from-the-source-tables)

The live Harbor task runs unconditioned monomers with an independent AF3 and Foldseek verifier. It has three locally measured baselines. The four-suite curves use the authors’ run records. An independent end-to-end reproduction of those four suites remains outstanding. Table S28 is an additional strict fixed-budget comparison. [S28 results](environment/references/s28/README.md)

[论文复现交接与待办（中文）](docs/REPRODUCTION_HANDOFF.zh-CN.md)

## Validation and test access

The agent sees the published suite-level baselines and curves. The operator retains the per-target archive and scores frozen submissions. The upstream validation configs overlap the Figure 2 targets. The fixed-target protocol allows policy selection on development runs using these targets. Final comparisons use frozen policies and fresh draws on the same roster. An additional family-disjoint panel is optional for studying generalization. The current live monomer verifier holds out random draws at the three public lengths. [Validation design and upstream overlap audit](docs/VALIDATION.md)

## Data

Three upstream annotated inputs encode chain lengths. The materializer fetches and verifies the public RFD4 checkpoint, official AF3 parameters and the Foldseek executable. Downloads total about 2.8 GB. The verifier additionally sees a 32-byte private seed. The public and private problems use the same lengths. [Materialization instructions](environment/data/MATERIALIZE.md)

## Workspace

The agent receives the frozen assets, development evaluator, executable starter, three runnable reference policies, their development and 150-slot results, and a writable research directory. It starts from stochastic TSR and can replace the inference code. [Stochastic TSR](environment/baseline/stochastic-tsr/RESULTS.md) · [Width-one greedy branching](environment/baseline/geometry-beam/RESULTS.md) · [Two-wide beam](environment/baseline/two-wide-beam/RESULTS.md)

## Evaluation

Executable submissions receive one versioned request at a time through `infer.py --request FILE --output DIRECTORY`. The runtime supplies the input, output count, generation seed, frozen model and budget. The default paper-size test retains the original 100-, 150- and 250-residue cohorts. A controller-supplied request roster supports other monomer lengths through the same entry point. [Program contract](environment/contract.md)

The verifier replays the selected human baseline on the same GPU and requests as each submitted program. The comparison reports both quality and measured cost, including final AF3, alongside the original clustered-pass reward. Reference choices include the reproduced stochastic TSR, width-one branching, two-wide beam and native Low and High. Varying the budget yields additional measured quality–throughput points. The archived run records are the standard historical comparisons. [Build an executable copy of a baseline](environment/evaluation/reference.py) · [Starter program](environment/baseline/program/infer.py) · [Export the measured curve](environment/evaluation/cost_curve.py)

The executable stochastic TSR replay at 100 residues matched the original public 10-slot cohort with eight AF3 passes and five passing clusters. A separate GPU test passed a 117-residue request through both isolated programs and their independent final scorer. [GPU check record](docs/program-reference-check.json)

The four-suite evaluator is `environment/references/overall/benchmark_curve.py`. Figure 2 uses generation attempts per GPU-hour on the horizontal axis. An alternate view includes final AF3 time. Report unique passing clusters per end-to-end GPU-hour separately for each suite. [Numerical results](environment/references/overall/operating_points.csv)

### Live monomer verification


Final evaluation draws 50 output slots at each length. The default generation/search limit is 1,800 elapsed seconds per length on one GPU. Legacy policy submissions additionally have the original 9,000 trunk-pass allowance per length. Executable programs are timed around their whole invocation, including discarded work and any internal scoring. The independent final refolder runs once on every valid finalist.

For each length, the verifier counts the Foldseek clusters that contain at least one sample with AF3 backbone scRMSD ≤ 1.5 Å. It divides by 50, including failures and empty slots. Reward is the mean of those three fractions. The result includes ordinary pass rate, emitted and refolded counts, generation time, AF3 time and unique passing clusters per allocated GPU-hour. The legacy policy evaluator also reports reserved trunk passes. Timing includes process startup and compilation. Repeated hits in one fold earn one point. [Contract](environment/contract.md) · [Scientific basis](environment/references/README.md)

## Running

Use the standard runner environment from the repository root.

```bash
runner/prepare.sh rfd4-inference-search /data/rfd4
export MOUNT_ROOTS="${MOUNT_ROOTS:-} /data/rfd4"
AGENT_KIND=codex runner/run.sh rfd4-inference-search
```

The declared development allocation is one H100 for 48 hours. The separate verifier uses one H100. To replay the executable stochastic TSR baseline on the private suite, package the pinned policy from the repository root into a fresh directory.

```bash
python tasks/rfd4-inference-search/environment/evaluation/reference.py stochastic-tsr /tmp/rfd4-reference
runner/run_verify.sh rfd4-baseline rfd4-inference-search /tmp/rfd4-reference
```

The public development set uses 30 slots. The public paper-size reproduction uses 150. Each release campaign must share a private seed across submissions. [Reproduction and score audit](docs/REPRODUCE.md)

## GPU fixture checks

The published seed-42 monomer agrees with a fresh local sample to 0.119 Å Cα RMSD, with 235 of 250 residue identities matching across hardware. Two independent runs of the released AF3-guided monomer beam made 48 internal AF3 evaluations each and agreed on all eight checkpoint rankings. A frozen gallery complex and an upstream crystal regression also exercised the released scoring runtime. These are direct GPU measurements on fixed examples. The four-suite operating points above remain numerical reconstructions from the authors’ archive. [Conformance results and replay](docs/gpu-pilot/README.md)

## Locally measured monomer results

| Policy | Development passing clusters / 30 | Paper-size AF3 passes / 150 | Paper-size passing clusters / 150 |
| --- | ---: | ---: | ---: |
| [Released stochastic TSR](environment/baseline/stochastic-tsr/RESULTS.md) | 20/30 | 123/150 | **82/150 (54.7%)** |
| [Width-one branching](environment/baseline/geometry-beam/RESULTS.md) | 21/30 | 112/150 | **71/150 (47.3%)** |
| [Two-wide beam](environment/baseline/two-wide-beam/RESULTS.md) | 11/30 | 101/150 | **46/150 (30.7%)** |

These are paired public experiments at 27,000 allocated trunk passes per full policy. Use the paper-size column for a comparison at the S15 sample count. Each length in the two-wide reproduction ran independently, so its timing diagnostic includes three AF3 startups. The production verifier runs the three lengths on a single device with one AF3 process.

In a second campaign, all five S15 samplers ran on the released checkpoint with the same three monomer lengths, public seeds, AF3 and Foldseek. The stochastic TSR repeat produced 73/150 passing clusters (48.7%) versus 82/150 (54.7%) above and 34.7% in the paper. ODE TSR produced 70/150 (46.7%). The report includes the timings and all 750 slot measurements. The 400-step arm exceeds the live task's compute cap. [Five-sampler protocol](docs/s15/README.md) · [Operating points](environment/references/s15/summary.csv) · [Local quality and throughput](environment/references/s15/quality-time.png)

The released autoencoder also completed reconstruction on all 640 official temporal-holdout structures. These measurements assess the autoencoder component. Inference policy evaluation uses AF3 self-consistency and distinct passing clusters. [AE holdout results](docs/ae-holdout/README.md#local-gpu-results)

## Limitations

This task measures inference for unconditioned, canonical single-chain monomers. A passing fold is a computational result. A private replay changes the sampling seeds. Table S15 does not identify every historical seed or runtime flag, so a new run can differ from its printed values. Fifty slots per length give a finite-sample comparison. A public tuning result should be interpreted alongside held-out replay. A trunk pass is a reproducible accounting unit at a fixed length. Actual GPU time also depends on batching and the cost of decoding. Timing comparisons require matched hardware and software.

## Citation and licenses

Cite Corley, Didi, Kazman, Brent et al. for RFD4-Proteina, Abramson et al. for AlphaFold 3 and van Kempen et al. for Foldseek. The RFD4 source, monomer inputs and model are released with the upstream BSD 3-clause license. Foldseek is GPL-3.0. This task fetches each model and the Foldseek binary from its publisher. Google’s model parameter and output terms apply to AF3. The repository distributes the task and aggregate scores. [References and code provenance](environment/references/README.md) · [Google model terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) · [Google output terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
