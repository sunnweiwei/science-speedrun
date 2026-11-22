# Validation for inference policy research

## What upstream provides

Figure 2 and supplementary §S8.1 use four public cohorts with 24, 77, 37 and 3 targets. We found no separately designated, target-disjoint development panel for selecting inference policies on all four suites. The model does have validation datasets for other purposes. Figure 3d evaluates held-out ligand stereochemistry, and the data release supplies a 640-structure temporal holdout for the autoencoder and a separate nanobody validation set. The 640 structures evaluate reconstruction by the autoencoder. The Figure 2 / S29 inference curves require the four design cohorts and their task-specific pass decisions. AE validation can be an additional component check. The authors also say that benchmark results informed their recommended recycling presets. [Supplementary §§S8.1 and S10.4](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true) · [Data release](https://huggingface.co/datasets/nvidia/RFD4-Proteina#benchmarks-and-project-media) · [Preset selection](https://rfd4-proteina.org/latest/how-to/run-the-model.html#recycling-schedules)

The following public configs directly reuse official target identities. Matching ignores case, the PPI roster's `rfd3_` prefix and the DNA input variant suffix. Counts measure target identity across conditions. Family-level overlap can be higher.

| Upstream validation config | Listed | Matched Figure 2 target IDs | Figure 2 suite size |
| --- | ---: | ---: | ---: |
| Protein backbone subset | 12 | 12 | 24 |
| MCSA subset | 15 | 15 | 37 |
| DISCO small subset | 12 | 10 | 77 |
| DISCO Figure 2c | 77 | 77 | 77 |
| DNA sequence-context | 3 | 3 | 3 |

Recheck these counts from the task directory with `python docs/check_validation_overlap.py /path/to/RFD4-Proteina`. The script reads upstream Git revision `6306a210dd17dcae8e4e0f381b5b74a719d9b30e` and the same official roster used for the reference curves. [Upstream configs](https://github.com/RosettaCommons/RFD4-Proteina/tree/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/configs/datasets/val)

## Fixed-target evaluation

In feedback relayed by the benchmark maintainers, a paper author suggested working directly on the benchmark set. We adopt the full 141-target Figure 2 roster as the task definition. The question is how well a policy performs on these known targets at a stated cost. The separation of development and final draws below is our evaluation design.

- **Development.** The agent may use development results on the public target roster to select a policy and its quality–throughput operating points. Keep the same development seeds across methods where possible. Small pilots can screen ideas. Label their sample counts because the clustered pass rate depends on the number of attempts.
- **Final comparison.** Freeze the policy package, operating points and compute accounting before evaluation. The controller uses independent random seeds on the same targets, with 200 fresh original attempts per target and method at each reported point. Score the baseline and candidate under matched conditions. Count failures, search costs and final AF3 costs using the official conventions. Keep final results outside the agent's selection loop. When reporting variability across repeats, compute each repeat's curve before summarizing across repeats.
- **Interpretation.** A gain on independent final draws supports a gain on the official fixed targets. A separate claim about unseen target families calls for an additional family-disjoint cohort. Such a cohort is optional for the fixed-target benchmark.

## Implementation status

The frozen autoencoder passed through the released reconstruction evaluator on all 640 temporal-holdout structures on an RTX A6000. The reported metrics group supervised residues across the full panel. Policy selection for monomer inference uses AF3 self-consistency and structural clustering on independent sampling draws. [AE results](ae-holdout/README.md#local-gpu-results)

The development image supplies historical suite-level scores, independent local RTX A6000 Figure 2 baselines (`rfd4-a6000-warm-v1`), and the live paired four-track evaluator (`evaluation/evaluate_tracks.py`). Official Harbor verification (`tests/test.sh`) evaluates unified `infer.py` packages across all four conditioned tracks on a held-out target/seed panel (`--panel verify` or `--panel full` with `/private/seed`) against same-GPU paired baselines (`native-low`, `native-medium`, `native-high`) at identical attempt counts and timing boundaries (`rfd4-paired-program-v1`), reporting flat numeric metrics in `/logs/verifier/reward.json` and multi-seed repeat SDs when `--repeats >= 2`.
