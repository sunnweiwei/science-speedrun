# Validation for inference policy research

## What upstream provides

Figure 2 and supplementary §S8.1 use four public cohorts with 24, 77, 37 and 3 targets. We found no separately designated, target-disjoint development panel for selecting inference policies on all four suites. The model does have validation datasets for other purposes. Figure 3d evaluates held-out ligand stereochemistry, and the data release supplies a 640-structure temporal holdout for the autoencoder and a separate nanobody validation set. Their endpoints differ from the four-suite quality–throughput evaluation. The authors also say that benchmark results informed their recommended recycling presets. [Supplementary §§S8.1 and S10.4](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true) · [Data release](https://huggingface.co/datasets/nvidia/RFD4-Proteina#benchmarks-and-project-media) · [Preset selection](https://rfd4-proteina.org/latest/how-to/run-the-model.html#recycling-schedules)

The following public configs directly reuse official target identities. Matching ignores case, the PPI roster's `rfd3_` prefix and the DNA input variant suffix. Counts measure target identity across conditions. Family-level overlap can be higher.

| Upstream validation config | Listed | Matched Figure 2 target IDs | Figure 2 suite size |
| --- | ---: | ---: | ---: |
| Protein backbone subset | 12 | 12 | 24 |
| MCSA subset | 15 | 15 | 37 |
| DISCO small subset | 12 | 10 | 77 |
| DISCO Figure 2c | 77 | 77 | 77 |
| DNA sequence-context | 3 | 3 | 3 |

Recheck these counts from the task directory with `python docs/check_validation_overlap.py /path/to/RFD4-Proteina`. The script reads upstream Git revision `6306a210dd17dcae8e4e0f381b5b74a719d9b30e` and the same official roster used for the reference curves. [Upstream configs](https://github.com/RosettaCommons/RFD4-Proteina/tree/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/configs/datasets/val)

## Proposed four-suite protocol

| Use | Cases and access | Decision |
| --- | --- | --- |
| Development / validation | A separately sourced panel in each suite. The agent may see its inputs, scores and curves. | Repeated policy comparisons and selection of operating points. |
| Published comparison | All 141 original Figure 2 targets. Only historical suite aggregates are included with the agent's references. | After the policy and operating points are frozen, the controller scores the submission and reports curves directly comparable to Figure 2 / S29. |
| Private generalization test | An additional, unpublished panel of disjoint target families with private seeds, held by the controller. | Estimate performance on unseen families. Releasing results after the campaign prevents test feedback from selecting the submitted policy. |

Choose development and private cohorts before observing candidate results. Match the official suites' inclusion criteria and distribution of target characteristics. Group aliases, alternative input conditions, related protein families, related chemical scaffolds and shared complexes as applicable. Audit overlap across suites as well as within each suite. Document independent family counts, provenance and any known overlap with model training data. For DNA there are only three official targets. Keep those together for the public comparison and source other groups for development and private evaluation.

Use the same pass decisions, within-target clustering and GPU accounting on all panels. Measure baselines and candidates on the same validation targets, hardware and paired random seeds. A quick screen can use a small subset of development families. For a paper-comparable curve, use 200 original attempts per target and method, count failures and report the same target-averaged yield. A clustered rate from a smaller sample can change simply because fewer duplicate folds have been seen. Estimate uncertainty across targets or families and paired repeated runs. A new seed on a familiar target measures sampling variation.

If only the existing public target records are available, grouped cross-validation can serve as an internal sensitivity analysis. Label the result with the reduced roster and keep it separate from the published full-panel score. With three DNA targets, report target-level variability and defer strong generalization claims until independent cases are available.

Fix the selected policy package hash, resource settings and list of operating points before the controller evaluates the official or private cohorts. Keep candidate test scores out of the iterative research loop. Historical official scores are already public and were visible in earlier task versions. Treat the full official panel as a public reporting benchmark. Use the additional sealed panel for an unseen-target claim.

## Implementation status

The development image contains the historical suite-level scores and plots, the locally measured monomer baselines and the upstream monomer example. The operator's checkout and verifier retain the official per-target archive and four-suite scorer. Published benchmark inputs can still exist in upstream software and remain publicly available.

The current live Harbor monomer task has three public lengths, a public development seed and a separate verifier seed. It measures transfer across draws at those lengths. The four-suite disjoint development and private panels described above have not yet been assembled or measured. The existing official curve scorer checks the exact published roster.
