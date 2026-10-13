# SOTA, published comparisons, and task-owned results

Checked on **2026-09-13**. This note separates current publication status,
evaluation protocol, and locally reproduced scores. No new model was trained
or selected during this literature check.

## Conclusion

**GLACIER with contrastive finetuning is the strongest published forward-model
mass-retrieval result located for the released MassSpecGym simulation domain:
69.95% Hit@1; its no-CF variant reports 69.69%.** This is a scoped literature
SOTA comparison, not a score on this task's complete 17,556-query test set.
The public source restricts the GLACIER recipe to `[M+H]+`, known collision
energy and supported instruments; our ID audit finds exactly 9,954 test queries.
See [paper Table 1](https://arxiv.org/html/2606.29161v1#S4.T1),
[current paper status](https://arxiv.org/abs/2606.29161), and the
[pinned source](https://github.com/coleygroup/ms-pred/tree/ed8311f22958cb37f055b663b5f56c5c77a2ee33).

For the complete task contract, **49.390522% is our strongest reproduced
reference**, obtained with official no-CF GLACIER and independently trained
clean MIST fallback. **40.322397% is the independently trained GLACIER +
DeepSets baseline.** These are aggregate exact-candidate test scores, not a
claim of community-wide first place or a matched-H100 comparison. The score
target remains the frozen `0.4939052175894281`.

**SpecBridge's 84.695% is not a current SOTA result:** the author withdrew the
paper on **2026-03-03**, citing a problem in the preprocessing/evaluation
pipeline. The latest arXiv record is v3, withdrawn; reading only the still
accessible v1 HTML misses this fact. Its advertised GitHub repository also
returned HTTP 404 during this check. Exclude the old result from active
comparisons; do not infer a corrected SpecBridge score from another model's
audit result. [Author's withdrawal notice](https://arxiv.org/abs/2601.17204v3).

## Other apparent SOTA candidates and their protocols

The rows below are literature context, not a shared leaderboard. A higher
number does not establish a better model when splits, candidate pools,
formula access, coverage or initialization differ.

| Method / source | Published Hit@1 | Protocol and disposition |
| --- | ---: | --- |
| GLACIER + CF / no CF | 69.95% / 69.69% | Released simulator domain, mass candidates capped at 256; scoped SOTA above. Our available official checkpoint is no-CF. |
| ICEBERG 2.0 in GLACIER Table 1 | 63.95% | Same reported simulator-domain mass comparison; predecessor to GLACIER. |
| [GLMR, Table 1](https://arxiv.org/html/2511.06259v1#S4.T1) | 64.172% mass / 68.478% formula | Non-withdrawn, AAAI 2026 paper; uses a pretrained ChemFormer pipeline and the older MassSpecGym protocol. A substantial full-task literature claim, but not a verified result on our pinned v1.5 inputs. Do not silently omit it or label it withdrawn. |
| [FLARE](https://pubmed.ncbi.nlm.nih.gov/41659479/) | 43.15% mass / 22.66% formula | Paper-reported tracks; we have not independently replayed its preprocessing, formula inputs and candidate contract on this task. |
| [MSAlign, Table 3](https://arxiv.org/html/2605.19752v1#S5.T3) | 16.2% / 32.3% | Original MCES split: respectively without true formula and with formula filtering. Uses pretrained DreaMS/ChemBERTa. |
| MSAlign, same table | 53.8% / 61.3% | A different **formula-disjoint data split**, respectively without/with formula filtering; not our official MCES split. |
| [MIST vFRIGID in the v1.5 audit, Table 3](https://arxiv.org/html/2606.19624v1#S5.T3) | 53.76% | Full **formula-known bonus** task; cannot substitute for formula-free mass retrieval. |
| [AIMe, August 2026](https://www.biorxiv.org/content/10.64898/2026.08.05.743095v1.full) | Not compared numerically | NIST-trained external evaluation; filters MassSpecGym targets for training-set similarity and uses at most 50 candidates selected with a different procedure. Not our fixed 17,556-query/256-candidate contract. |
| SpecBridge v1 | 84.695% | Withdrawn; excluded as described above. |

The [official repository's retrieval CSV](https://github.com/pluskal-lab/MassSpecGym/blob/6396b90feb5785f0ac30497c1ff4d7ce5b580f2c/results/retrieval.csv)
contains seven older rows, with JESTR at 15.13% as its maximum. The last change
to that file is commit `6396b90` (2025-10-11). It omits the 2026 methods and
cannot settle the current SOTA. Searches covered current primary records,
the v1.5 audit, the official source/results, and newer MassSpecGym retrieval
papers through the check date. No reviewed source establishes one complete,
current v1.5 mass-retrieval ranking under our exact input and scoring contract.
Absence of such a ranking does not prove that our reference is the best method.

## Why v1.5 verification matters

The [MassSpecGym audit, Section 5.1](https://arxiv.org/html/2606.19624v1#S5.SS1)
shows that inconsistent SMILES formatting can identify the answer independently
of spectral evidence. Its DreaMS/ChemBERTa alignment experiment falls from
82.41% to 6.70% after correction **on the formula-known task**. These are that
audit's model and conditions, not an exact remeasurement of SpecBridge or GLMR.
The same audit distinguishes formula leakage, pretraining overlap and metric
differences. Consequently we require a valid current publication, the same
split/canonicalization/candidate inputs, no true test formula, full coverage,
and the declared initialization policy before calling an external number a
comparable task result.

## Reproduced results on the fixed task

All full-test rows below contain 17,556 valid rankings. Percentages are Hit@k;
MCES is raw distance, lower is better. Seeds/recipes differ across methods.

| Method | Initialization / data actually used | Hit@1 | Hit@5 | Hit@20 | MCES@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Official GLACIER no-CF + clean MIST | Author's GLACIER checkpoint; MIST trained on experimental train + clean synthetic data | **49.390522%** | 64.188881% | 74.436090% | 12.178913 |
| Scratch GLACIER + DeepSets | Random weights; train-only MAGMa supervision for GLACIER; experimental train only for DeepSets | **40.322397%** | 56.846662% | 70.773525% | 13.169600 |
| Clean MIST alone | Independently trained encoder; inferred top-five formulas, no true test formula | 39.872408% | 51.486671% | 63.140807% | 15.166838 |
| Repaired DeepSets starter, acceptance run | Random weights; experimental train only | 4.004329% | 12.354751% | 27.403737% | 22.559581 |

The earlier DeepSets result of 5.126% belongs to a separate September 2 run,
not the weights used in this scratch hybrid. Likewise 35.987441% is the
scratch hybrid's **validation** Hit@1 over 19,429 queries, not its test score.

| Test branch | Queries | Official-weight hybrid Hit@1 | Scratch hybrid Hit@1 | Contribution to full-score gap |
| --- | ---: | ---: | ---: | ---: |
| GLACIER-supported domain | 9,954 | 67.490456% | 65.541491% | 1.105035 percentage points |
| Remaining conditions | 7,602 | 25.690608% (MIST) | 7.300710% (DeepSets) | 7.963090 percentage points |

The fallback explains 87.81% of the 9.068125-point arithmetic gap. This
localizes the difference; it does not causally separate model architecture,
augmentation, training budget or random initialization. No test-driven model
change was made. Reproduction evidence is in
[reference_baseline.json](reference_baseline.json),
[task_trained_baseline.json](task_trained_baseline.json), and the operator
[release ledger](../../docs/release_validation.json).
