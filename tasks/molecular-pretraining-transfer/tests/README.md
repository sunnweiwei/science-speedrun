# Verifier

Isolated two-container offline verifier (`tests/` is never copied or mounted into the agent development image `/workspace`, as verified in `environment/Dockerfile` and `environment/docker-compose.yaml`).

| file | role |
|---|---|
| `docker-compose.yaml` | `main` (GPUs, no network): fine-tune data from `TASK_DATA_DIR/public/downstream`, test molecules without labels from `TASK_PRIVATE_DIR/verifier/test-inputs`. `grader` (no GPU, no network): labels from `TASK_PRIVATE_DIR/private/labels`. The two share the `sealed` volume. |
| `test.sh` → `main_verifier.py` | validates and freezes `/workspace/submission`, then runs `protocol/check_encoder.py` and the 24 fine-tunes (`protocol/run_suite.py`) as the unprivileged `agent` user. It seals the predictions (hashes + READY) or writes `score: 0.0, reward: 0.0, valid: 0.0` and INVALID, then copies the grader's `reward.json` / `report.json` to `/logs/verifier`. |
| `grader.py` | checks the snapshot and scores each task against the labels (`protocol/tasks.py`), outputting `score` and `reward` = `1 / (1 + mean_4_groups(MA_RAE))`; never runs package code |
| `protocol/` | byte-identical to `environment/evaluation/protocol/` (checked by `test_task.py`) |
| `test_task.py`, `test_encoder_checkpoint.py`, `run_unit_tests.sh` | CPU unit tests |

A replay outside Harbor needs an override that mounts a package at `/workspace/submission` and a host directory at
`/logs/verifier`; `docs/VALIDATION.md` records the acceptance replays.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary task score (`reward.json["score"]` and `reward.json["reward"]`, bounded in $[0, 1]$) measures low-data molecular transfer across 24 downstream regression tasks spanning **4 equally weighted (`1/4` each) benchmark groups** (`qm9`, `compas1d`, `expansionrx`, `asap`):

$$\overline{\mathrm{MA\text{-}RAE}} = \frac{1}{4}\Big(\mathrm{MA\text{-}RAE}_{\text{qm9}} + \mathrm{MA\text{-}RAE}_{\text{compas1d}} + \mathrm{MA\text{-}RAE}_{\text{expansionrx}} + \mathrm{MA\text{-}RAE}_{\text{asap}}\Big), \qquad S_{\text{molecule}} = \frac{1}{1 + \overline{\mathrm{MA\text{-}RAE}}}$$

where for each single-target task $t$ (and averaged across HOMO, LUMO, gap for the 3-target `qm9dft_v2` task, matching Uni-Mol2's `agg_mae`), Relative Absolute Error is $\mathrm{RAE}_t = \mathrm{MAE}_t / \frac{1}{n}\sum_i |y_i - \bar{y}|$, and group $\mathrm{MA\text{-}RAE}_g$ is the arithmetic mean of $\mathrm{RAE}_t$ over the tasks in group $g$.

Every component of this protocol directly inherits community-standard upstream benchmarks:
- **3D Molecular Pretraining & Quantum Transfer (`QM9` and `COMPAS-1D`)**: Following *Uni-Mol2* ("Uni-Mol2: Exploring Molecular Pretraining Model at Scale", Ji et al., *NeurIPS* 2024 / `arXiv:2406.14969`, Tables 4–6), 3D encoder representations are evaluated on *QM9* DFT electronic/thermodynamic properties (Ramakrishnan et al., *Scientific Data* 2014) and *COMPAS-1D* polycyclic benzenoid aromatic properties (Wahab et al., *J. Chem. Inf. Model.* 2022) under scaffold-similarity splits with InChIKey-connectivity pretraining leakage removal.
- **Prospective Medicinal-Chemistry ADMET Transfer (`OpenADMET-ExpansionRx` and `Polaris ASAP-Discovery`)**: Both real-world drug-discovery campaigns—the *OpenADMET-ExpansionRx Blind Challenge* (2025–2026, 9 temporal-split endpoints) and the *Polaris ASAP-Discovery Antiviral ADMET Challenge* (2025, 5 endpoints)—establish **Macro-Averaged Relative Absolute Error (`MA-RAE`)** in $\log_{10}(\max(y, 0) + 1)$ space (with `LogD` unlogged) as their official primary leaderboard metric because $\mathrm{RAE}_t$ normalizes heterogeneous physical units (Hartree, Debye, Bohr$^3$, $\text{eV}$, $\mu\text{M}$, $\text{mL/min/kg}$) onto a common scale where predicting the test-set mean achieves exact $\mathrm{RAE} = 1.0$.
- **Monotone Bounded Reciprocal Mapping (`1 / (1 + mean_4_groups(MA_RAE))`)**: Because $\mathrm{MA\text{-}RAE} \in [0, \infty)$ is an unbounded non-negative error ratio, $S = 1 / (1 + \overline{\mathrm{MA\text{-}RAE}})$ maps $[0, \infty)$ smoothly and strictly monotonically onto $(0, 1]$ without arbitrary upper-error clipping cliffs: a trivial mean predictor ($\overline{\mathrm{MA\text{-}RAE}} = 1.0$) lands at exact `0.5`, unpretrained random initialization (`0.867`) scores `0.536`, the 30k-step 84M starter (`0.432`) scores `0.698`, released full-schedule Uni-Mol2 84M (`0.406`) scores `0.711`, perfect zero error approaches `1.0`, and invalid/failed submissions receive `0.0`.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **Why Equal 4-Group Weighting (`1/4` Each) Is Used Instead of Flat 24-Task Averaging**: The four benchmark sources contain unequal numbers of fine-tune tasks (`ExpansionRx`: 9, `QM9`: 6, `ASAP`: 5, `COMPAS-1D`: 4). A flat 24-task average would weight ExpansionRx $2.25\times$ more heavily than COMPAS-1D; equal `1/4` group averaging gives balanced `50% / 50%` weight to quantum-chemical structure-property transfer (`QM9 + COMPAS-1D`) and experimental ADMET transfer (`ExpansionRx + ASAP`).
2. **Why `Spearman`, `Kendall`, and `R^2` Are Not Mixed into the Primary Score**: Rank correlations (`Spearman`, `Kendall` $\tau_b$) are invariant to shift and scale calibration errors (a prediction shifted by $100\times$ still gets Spearman `1.0`), whereas `R^2` squares outlier residuals and becomes heavily skewed on small-sample noisy ADMET assays (`ASAP` $n \approx 110$). Following the official *OpenADMET-ExpansionRx* and *Polaris ASAP* challenge leaderboards, `MA-RAE` is the sole primary metric, while rank and variance-explained metrics are preserved as diagnostics.
3. **Why Fine-Tuning Hyperparameters Are Frozen Rather Than Submission-Controlled**: Allowing a submission to customize per-task learning rates, epochs, ensembling, or task-specific heads would confound downstream hyperparameter search with pretrained representation quality. Freezing the 15-epoch Uni-Mol2 fine-tuning protocol isolates the scientific variable of interest—the pretrained encoder (`<= 90M` encoder parameters, verified full state-dict coverage).

### 3. Diagnostic Signals Retained in report.json

To provide comprehensive visibility into transfer behavior across chemical regimes without complicating the scalar objective, `report.json` (and flat keys in `reward.json`) retains:
- **Group-level summaries (`report["groups"]`)**: `qm9_ma_rae`, `compas1d_ma_rae`, `expansionrx_ma_rae`, `asap_ma_rae` alongside `qm9_mean_spearman`, `compas1d_mean_spearman`, `expansionrx_mean_spearman`, and `asap_mean_spearman`.
- **All 24 per-task metrics (`report["tasks"]`)**: exact test-set `mae`, `rae`, `r2`, `spearman`, `kendall`, and molecule count `n` for every quantum-chemical and ADMET endpoint.
- **Integrity & compute provenance**: `encoder_check` (verified encoder architecture, non-head parameter count against the 90M cap, and strict state-dict key coverage) and `fine_tunes` (`suite.json` per-task fine-tuning wall-clock durations and retry counts).
