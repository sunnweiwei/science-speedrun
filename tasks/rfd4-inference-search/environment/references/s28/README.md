# Equal-cost reference from Table S28

The selected paper reference is the ten-target strict interaction experiment in supplementary §S7.1.3 of Corley et al. Each method receives four GPU-hours per target, or 40 GPU-hours for its ten targets. The outcome is the number of distinct structural clusters containing a computational passer. Independent sampling applies the conditions at the end. Interaction search also uses an intermediate score to stop unpromising trajectories. The stopped attempts consume budget. Table S28 compares these two methods. It gives no separate measurement for adding a wider beam to early stopping. [Supplementary methods, §S7.1.3 and Table S28](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true)

| Official method | Passing designs | Passing clusters | Clusters / allocated GPU-hour |
| --- | ---: | ---: | ---: |
| Independent sampling | 1,398 | 541 | 13.53 |
| Interaction-guided early stopping | 2,342 | 674 | 16.85 |

The gain is 133 clusters, or 24.6% relative to independent sampling. The calculation uses the integer counts before rounding. Early stopping wins on seven targets, ties on one and loses on two. SAM has zero passing clusters in both arms. The observed yield on that target identifies an unresolved case. It provides no empirical winner there. [Ten published target rows](target_counts.csv)

## Recheck the paper figures

The CSV contains the four count columns extracted from the v1 supplement. The source is the page numbered 59 in the manuscript (page 34 of the supplementary PDF). Run from the task directory.

```bash
python environment/references/s28/audit.py
uv run --no-project --with pymupdf==2.1 python environment/references/s28/audit.py \
  --supplement /path/to/RFD4-Proteina-v1-supplement.pdf
```

The first command recomputes the totals and rates from the target counts with Python's standard library. The second checks the SHA-256 of the source PDF, reads the rows directly, checks each printed rate and checks the pooled row. The results reproduce the arithmetic of the published table. The PDF provides target-level counts and the allocated hours. They do not establish the results of a fresh model run. The existing Harbor monomer score belongs to Table S15's problem family. [Live monomer measurements](../../../README.md)

The v1 supplementary-materials page lists one PDF. In the official model release and benchmark website we found target-level exports for the standard benchmarks. We did not locate the S28 attempt-level scores, cluster assignments or timing records there. The standard 77-target small-molecule benchmark uses a different cohort and endpoint. [Supplementary files](https://www.biorxiv.org/content/10.64898/2026.11.12.766037v1.supplementary-material) · [Official released benchmark files](https://huggingface.co/nvidia/RFD4-Proteina/tree/5ec66c399bf0b48727ab24e6b298dcf33864fa07/benchmarks)

## What would establish a new result

Use the same ten targets, per-target compute allowance, final criteria and clustering. Independently rerun both published arms, recording all attempts, independent pass and cluster assignments, costs for stopped attempts and all model and oracle time. Counts from fresh random draws can vary. Assess agreement across predeclared seeds or repeated runs, with the same hardware and accounting for every arm.

A proposed beam method would be a third arm. Report its gain against the stronger published control, together with per-target results on hard cases and uncertainty across matched repeats. A nonzero result on a previously unsolved target and a reproducible increase in pooled yield answer different questions.

For comparison, Table S18 has a different difficult case and 24 GPU-hours per method. It reports nine unique passers for temporal search, three for independent generation and zero for the constant-criteria search. Table S29 reports that structure-predictor-guided search improves the clustered pass percentage on its four standard benchmarks, while the high-thinking baseline has greater clustered yield per GPU-hour including final AF3 in each comparison. These results have different tasks, budgets and endpoints. [Tables S18, S28 and S29](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true)
