# Scientific metric and validation protocol

Status: adopted as `drugclip-dataset-bedroc/v2` in task version 0.3.0.
The grader uses this definition; historical v1 reward files retain their original
values and can be migrated into new, explicitly versioned reports.

## Scientific objective and evidence

The task is early retrieval: given a protein binding pocket and a large
candidate library, rank genuine binders near the top for experimental testing.
A useful score must be sensitive to improvements in these early ranks.

- DrugCLIP, *Science* (2026), evaluates DUD-E and LIT-PCBA primarily with EF1%.
  The [authors' manuscript](https://www.biorxiv.org/content/10.1101/2024.09.02.610777v3.full)
  describes this setup; the [published article](https://doi.org/10.1126/science.ads9530)
  and [official metric code](https://github.com/THU-ATOM/Drug-The-Whole-Genome/blob/9d0db73915219763b5bcc3b7de86d7c4edb5ed03/unimol/tasks/drugclip.py)
  anchor comparisons. The official code separately calculates ROC-AUC,
  BEDROC(alpha=80.5), and several EF cutoffs, without `E/(1+E)`.
- [AdaMBind, *Nature Communications* (2026)](https://www.nature.com/articles/s41467-026-70554-5)
  evaluates LIT-PCBA early recognition using EF@1%, EF@5%, Precision@10,
  Precision@20, and BEDROC(alpha=80.5). It reports distinct measures rather
  than inventing an uncalibrated arithmetic mean of them.
- [Truchon and Bayly, *JCIM* (2007)](https://doi.org/10.1021/ci600426e)
  introduced BEDROC to address early recognition. It uses exponential rank
  weights and a best/worst ranking normalization to a 0–1 scale. It is not
  classification accuracy. Random performance is not universally 0 or 0.5.
- [Chen et al., *PLOS ONE* (2019)](https://doi.org/10.1371/journal.pone.0220113)
  demonstrate how DUD-E analogue and decoy bias can produce misleading
  screening performance, including strong models that omit receptor input.
  [LIT-PCBA, *JCIM* (2020)](https://doi.org/10.1021/acs.jcim.0c00155)
  uses experimental active/inactive compounds and addresses several such
  biases. It is not a guarantee against every form of overlap or shortcut.

## Primary reward

For every target, calculate standard BEDROC with alpha fixed to 80.5. Keep
the complete frozen candidate library; do not subsample easy negatives or
rebalance active prevalence. Average targets within each dataset, then give
the two datasets equal weight:

```
B_DUD = sum(BEDROC_80.5 for each of 102 DUD-E targets) / 102
B_LIT = sum(BEDROC_80.5 for each of 15 LIT-PCBA targets) / 15
reward_v2 = (B_DUD + B_LIT) / 2
```

This score considers the ranked list with emphasis on early retrieval, rather
than a single cutoff. It requires no further saturation mapping. The alpha
value is already used by our scorer and the cited evaluations. Dataset
balancing is our benchmark-design choice, not a published DrugCLIP score:
it prevents the 102-target artificial-decoy dataset from supplying 87% of
the overall weight. It does not eliminate dataset bias or replace reporting
each dataset separately. Weights and alpha must be fixed before new agent
comparisons, not optimized to produce a desired baseline score.

EF1% remains the main literature-facing companion metric. Also retain
EF0.5%, EF2%, EF5%, ROC-AUC, and per-target distributions, and report exact
Recall@1% for interpretation. These measures share information and have
different scales; directly averaging all six would overweight early retrieval
several times without a scientific justification. BEDROC covers early
retrieval more broadly, but no scalar captures every desirable behavior.

## Consequence for the already verified reference

Re-aggregation of the completed epoch-14 Harbor replay gives:

| Dataset | Targets | Mean BEDROC(80.5) | Expected random BEDROC |
| --- | ---: | ---: | ---: |
| DUD-E | 102 | 0.294588 | 0.022293 |
| LIT-PCBA | 15 | 0.038182 | 0.017352 |
| Equal-dataset overall (v2) | — | **0.166385** | **0.019822** |

The old equal-target BEDROC mean is 0.261715, while the old transformed EF
reward is 0.941548. They are different quantities. The new number is not
evidence that the benchmark has become harder: no predictions changed.
Actual difficulty and headroom must be judged from retrieval behavior and
generalization. On LIT-PCBA, this checkpoint has EF1% below random on 7 of
15 targets and zero top-1% hits on 4. Those results warrant per-target analysis.

The v2 overall's target-bootstrap 95% interval is [0.1381, 0.1959],
using 20,000 draws with seed 20261105, sampling targets independently within
each dataset. This describes between-target uncertainty, not training-seed
variation or laboratory efficacy. The [calculation receipt](metric_review.json)
identifies the source report by SHA256. Re-aggregation requires no retraining
or model inference when per-target BEDROC results are retained.

## Required protections and outstanding scientific checks

No metric is intrinsically unhackable. The score and evaluation protocol
must be assessed together.

Already tested: separate verifier; read-only development mounts; withheld
labels and index mappings; fresh private molecule orders; field-whitelisted
inference inputs; privilege drop; complete target/ligand coverage; finite
predictions; expected tie ranks; rejection of invalid submissions; constant
predictor controls. See [acceptance](ACCEPTANCE.md). Existing preparation
validates pinned protein/ligand filtering metadata and disjoint development
splits. Generic Uni-Mol pretraining overlap is not established by these checks.

Before making a stronger scientific robustness claim, additionally require:

1. A ligand-only control trained only on permitted data, and an evaluation
   with mismatched protein pockets. Measure the added value of correct pocket
   information; do not assume any fixed drop threshold proves or disproves
   cheating. Legitimate ligand signals can exist.
2. Train/dev/test overlap audits at protein-family, exact molecular identity,
   and molecular-scaffold levels, including provenance of allowed pretraining.
   Near-neighbor filtering alone does not prove scaffold disjointness.
3. Test-score feedback withheld during agent development. Keep a sealed final
   evaluation or additional unseen targets/assays for checking improvements
   selected after repeated public-benchmark comparisons. Public labels remain
   susceptible to memorization even if local files and networking are hidden.
4. Target-level paired comparisons and bootstrap intervals for agent versus
   baseline; inspect results on both datasets and failed targets. A gain on
   the aggregate alone is insufficient evidence of broad improvement.

These additional scientific controls have not yet been run in this audit.
They must not be described as passing or as a guarantee of no reward hacking.
This metric concerns virtual screening, not other drug-development
endpoints that this task does not measure.

## Migration

Task version 0.3.0 records `drugclip-dataset-bedroc/v2` in task.toml and every
verifier report. `tests/reaggregate.py` recomputes the primary reward from all
117 target records, preserves the source hash and legacy reward, and refuses
to overwrite existing files. Baseline, references and agent comparisons must
use the same aggregation. Partial reports have diagnostics but no primary
score. Old and new rewards must not share an axis as though they had the same
definition. The metric changes aggregation only; the ranking metric itself,
model weights, data and inference remain unchanged.
