# Independent verifier

Two-service replay, mirroring `clinical-trial-modeling`:

- **main** (`test.sh` → `main_verifier.py`): validates the submission package
  (requires `infer.sh` and `README.md`, rejects escaping symlinks), validates
  the verifier inputs against their manifest hashes, runs Pass 1
  (`bash infer.sh /benchmark PASS1_OUT`) on the canonical input manifest and
  Pass 2 (`bash infer.sh SWAPPED_PPI_IN PASS2_OUT`) on a read-only staged input
  manifest where `yeast_ppi` and `human_ppi` pairs are swapped to
  `(sequence_b, sequence_a)` in shuffled row order, each as a demoted `agent`
  user with an ephemeral `HOME`/`TMPDIR` and no network. It then copies only the
  eleven bounded `predictions/<task>.jsonl` files plus
  `yeast_ppi_swapped.jsonl` and `human_ppi_swapped.jsonl` into a read-only
  sealed snapshot. Labels are **not** mounted in this service.
- **grader** (`grader.py`): waits for the sealed snapshot, cross-checks
  snapshot and private-label manifests, scores every task with `metrics.py`
  (computing symmetric PPI probabilities `p_sym = 0.5 * (p_ab + p_ba)` for
  `yeast_ppi` and `human_ppi` and aggregating the 11 chance-calibrated task
  skills across the 4 equal-weight capability pillars), and writes `report.json`
  plus a flat `reward.json` (with both `"score"` and `"reward"`) to
  `/sealed/results`, which `main_verifier.py` copies to `/logs/verifier`.

Failure handling: a malformed, missing, duplicate, or unknown prediction file
zeros only that task (`skill = 0.0`) and records `invalid_reason`; setup
failures (bad submission package, input hash mismatch, snapshot mismatch) abort
the run and are therefore distinguishable from measured model performance.

Isolation boundary: inference runs in a container without the private labels
and cannot modify inputs (read-only mount), inspect Pass 1 outputs during Pass 2
(`0700` root-owned directory lock + fresh ephemeral `HOME`/`TMPDIR`), or alter
the sealed snapshot (created after inference exits, hash-pinned).

**The privilege drop is verified, not assumed.** `main_verifier.py` runs the
submission as the unprivileged `agent` user, and after `setuid` it checks that
the drop actually took effect and that root cannot be regained, aborting if
either fails. Under sandboxes that emulate root with an `LD_PRELOAD` shim rather
than a real uid mapping — apptainer's fallback `--fakeroot` when the user has no
`/etc/subuid` allocation, for example — `os.setuid()` returns success and
`geteuid()` reports the new uid while the process retains full privileges.
Without the post-condition check the verifier would run untrusted submission
code as root and look perfectly healthy doing it.

The verifier also refuses to start unless it is actually root, rather than
failing obscurely later, so there are only two outcomes: a real privilege drop,
or a clear refusal. It never silently degrades.

`metrics.py` is kept byte-identical with
`../environment/evaluation/metrics.py` so development self-scores match.

Unit tests (synthetic, no data needed): `bash run_unit_tests.sh`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The eleven evaluation tasks originate from the **PEER** multi-task protein
sequence understanding benchmark (**Xu et al.**, *"PEER: A Comprehensive and
Multi-Task Benchmark for Protein Sequence Understanding"*, *Advances in Neural
Information Processing Systems (NeurIPS) Datasets and Benchmarks*, 2022,
arXiv:2206.02096), building on the upstream TAPE (**Rao et al.**, *NeurIPS*
2019), DeepSol, DeepLoc, SCOPe remote homology, NetSurfP/CB513, STRING PPI, and
SKEMPI datasets.

Following PEER's taxonomy of protein sequence understanding into four core
biological capability families, the primary V4/V3 benchmark score (`score` /
`reward` in `reward.json`, identical to `harness/ssi_index_v3.py:sequence()`)
aggregates chance-calibrated per-task skills $v_k$ across four equal-weight
($1/4$) pillars:

1. **Function ($1/4$)**: mean of `fluorescence` ($\rho_s$), `stability`
   ($\rho_s$), `beta_lactamase` ($\rho_s$), and `solubility`
   ($\frac{\text{BA} - 1/2}{1 - 1/2}$).
2. **Localization ($1/4$)**: mean of `subcellular_localization`
   ($\frac{\text{BA} - 1/10}{1 - 1/10}$) and `binary_localization`
   ($\frac{\text{BA} - 1/2}{1 - 1/2}$).
3. **Structure ($1/4$)**: mean of `fold`
   ($\frac{\text{BA} - 1/136}{1 - 1/136}$) and `secondary_structure`
   ($\frac{\text{Acc} - 0.4306464618}{1 - 0.4306464618}$).
   - **Why `fold` uses $K = 136$**: Although `fold` outputs logits across all
     $1{,}195$ SCOPe fold classes (`CLASSES["fold"] = 1195`), the held-out
     `test_fold_holdout` split contains exactly $136$ observed fold classes with
     positive ground-truth support. Because `balanced_accuracy` averages recall
     strictly over classes present in ground truth ($\text{TP}_c + \text{FN}_c > 0$),
     any constant or uniform classifier achieves
     $\text{BA} = \frac{1}{136}\sum_{c=1}^{136} R_c = \frac{1}{136}$ on the held-out
     test split. Using $1/136$ therefore calibrates zero skill to exact empirical
     chance on the evaluation split, whereas $0.4306464618$ calibrates CB513
     3-state accuracy against the majority coil residue fraction.
4. **Protein interactions ($1/4$)**: mean of `yeast_ppi`
   ($\frac{\text{BA}_{\text{sym}} - 1/2}{1 - 1/2}$), `human_ppi`
   ($\frac{\text{BA}_{\text{sym}} - 1/2}{1 - 1/2}$), and `ppi_affinity`
   ($1 - \frac{\text{RMSE}}{3.083206134}$, calibrated against the constant-mean
   test RMSE anchor $3.083206134$).

For the undirected physical interaction tasks `yeast_ppi` and `human_ppi`,
pair-input benchmarks are notoriously vulnerable to ordered-pair artifacts and
hub/length biases (**Park & Marcotte**, *"Flaws in evaluation schemes for
pair-input computational predictions"*, *Nature Methods* 9, 1134–1136, 2012,
doi:`10.1038/nmeth.2259`). In upstream PEER `human_ppi`, positive pairs exhibit a
systematic ordering artifact where `len(sequence_b) > len(sequence_a)` by roughly
$100$ amino acids on average, allowing an asymmetric sequence-free heuristic such
as $p(A, B) = \sigma(|B| - |A|)$ to score far above chance on canonical `(A, B)`
inputs alone. Evaluating every undirected pair in both orientations `(A, B)` and
`(B, A)` in the label-free `main` container and scoring the symmetrized
interaction probability $p_{\text{sym}} = \frac{1}{2}(p(A, B) + p(B, A))$
restores physical permutation invariance ($p_{\text{sym}}(A, B) = p_{\text{sym}}(B, A)$)
and collapses any pure input-order bias to exact chance ($p_{\text{sym}} = 0.5$,
$\text{BA}_{\text{sym}} = 0.5$, $v_k = 0.0$).

Finally, `signed_score` is the equal-weight mean of the four pillars and
`score = clip(signed_score, 0.0, 1.0)`.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No hard rejection gate on `|p_ab - p_ba|`**: Legitimate multi-task models
  using asymmetric pair heads — including the reference ESM-2 starter, which
  concatenates `[pool(sequence_a), pool(sequence_b)]` — naturally exhibit
  non-zero orientation differences $|p(A,B) - p(B,A)| > 0$ unless trained with
  explicit pair augmentation or symmetric pooling. Imposing an ad-hoc threshold
  gate on `asymmetry_mae` would arbitrarily disqualify standard baseline
  architectures, whereas post-hoc probability symmetrization
  $p_{\text{sym}} = \frac{1}{2}(p(A,B) + p(B,A))$ already removes 100% of any
  directional ordering shortcut mathematically.
- **No pre-pillar zero-clamping of individual task skills**: Individual task
  skills $v_k$ remain signed inside each capability pillar (clamped to $[0, 1]$
  only at the top-level four-pillar mean), matching `ssi_index_v3.sequence()` so
  anti-correlated or worse-than-constant predictions on one task cannot be masked
  for free while optimizing another task in the same pillar.
- **No ad-hoc parameter-count or latency penalties**: Within the fixed offline
  compute and wall-clock budget, PEER evaluates representational quality across
  diverse sequence tasks rather than enforcing artificial model-size caps.

### 3. Diagnostic Signals Retained in report.json

`report.json` retains full hierarchical diagnostics alongside `score` and
`reward`:

- `pillars`, `task_skills`, and `signed_score`: exact per-pillar and per-task
  signed V3 skill breakdown before top-level `[0, 1]` clipping.
- Native task metrics per PEER dataset: `accuracy`, `balanced_accuracy`, and
  `macro_f1` for classification tasks; `rmse` and `spearman` for `ppi_affinity`;
  `spearman` for fitness/stability regressions; and masked `accuracy` plus
  `valid_residues` for `secondary_structure`.
- Orientation-symmetry diagnostics on `yeast_ppi` and `human_ppi`:
  `p_ab_mean`, `p_ba_mean`, `p_sym_mean`, `asymmetry_mae`
  ($\frac{1}{N}\sum |p(A,B) - p(B,A)|$), single-orientation `ab_accuracy`,
  `ab_balanced_accuracy`, `ba_accuracy`, `ba_balanced_accuracy`, and per-example
  `pair_probabilities` (`p_ab`, `p_ba`, `p_sym`).
