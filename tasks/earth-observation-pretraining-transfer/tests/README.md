# Independent verifier

## Held-out AWF spatial blind evaluation & V4 default grading

By default, `tests/test.sh`, `tests/grade.py`, and `tests/grade_awf_knn.py` evaluate AWF using the frozen spatial roster at `tests/sealed/spatial-roster.json` (`roster_sha256: ed0cf3cebf2d417011b367b7616bc8e3df6950a18462ec52a46356b5ba457f86`), which can be overridden via `OLMO_AWF_KNN_ROSTER` or `--roster`. This roster partitions the 1,459 AWF windows into 605 connected 128-pixel spatial blocks (`pairs_across_splits == 0`): `train = 1,053` (431 blocks), `valid = 201` (89 blocks), and `test = 205` (85 blocks; 156 originally in upstream `train` and 49 in upstream `val`). During development materialization (`environment/evaluation/screening.py prepare` and `environment/data/prepare_hf.py`), the label-free allowlist `environment/data/awf_spatial_dev_splits.json` excludes all 205 blind `test` windows (both imagery and labels) from `/workspace/data/eval/awf/`.

The scorer applies the OlmoEarth paper's cosine-weighted 20-neighbour vote at temperature `0.07` to unstandardized frozen features, using the 1,053 spatial `train` windows as neighbours, scoring on the 205 spatial `test` windows, and reporting the 201 spatial `valid` windows as a diagnostic. Application raster codes are mapped to the class order used by the paper (`PAPER_CLASS = (8, 5, 6, 2, 1, 0, 4, 3, 7)`), and the scorer asserts zero window-ID overlap across `train`, `valid`, and `test` as well as exact agreement with `awf_spatial_dev_splits.json`. [Paper classifier and published features](https://huggingface.co/datasets/allenai/olmoearth-paper-embeddings/tree/6ea2c7973d2d6b79cd2221370bea6319f0eada68) · [Paper classes and splits](https://github.com/allenai/rslearn_projects/blob/157cf6f2f18ce849dfdc6872d85f298f936de331/data/olmoearth_evals/tasks/awf_base.yaml) · [Application raster classes](https://github.com/allenai/olmoearth_projects/blob/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a/olmoearth_run_data/awf/olmoearth_run.yaml)

For an AWF-only replay from saved features, run the following in the verifier runtime:

```bash
python tests/grade_awf_knn.py features --features features.npz --labels labels.json \
    --roster tests/sealed/spatial-roster.json --output awf-result.json
```

`OLMO_INFERENCE_HASHSEED=0 bash tests/test.sh` reads `OLMO_MODEL` (default `/workspace/submission`), `OLMO_EVAL_DATA` (default `/data/eval`), and writes `reward.json` (flat numeric fields including `score` and `reward`, as Harbor requires) and `report.json` (status, reasons, diagnostics, and SHA-256 hashes) to `OLMO_VERIFIER_OUTPUT` (default `/logs/verifier`).

For each task, the verifier reruns `infer.py` from the exported package inside `bubblewrap` with no network and no labels or training data mounted. It then scores AWF with the frozen cosine `20-NN` probe on the 205 spatial blind test windows and fits the standardized LFMC ridge probe on downstream training labels (selecting regularization `alpha` on validation before evaluating test **MAE**).

The primary scalar score (`reward.json["score"]` and `reward.json["reward"]`) matches `harness/ssi_index_v3.py:earth()` identically:

```text
awf_skill = (awf_test_accuracy - 59/205) / (1 - 59/205)
lfmc_skill = 1.0 - lfmc_test_mae / 24.435180349813233
signed_score = 0.5 * awf_skill + 0.5 * lfmc_skill
score = clip(signed_score, 0.0, 1.0)
```

Malformed packages and predictions produce `score: 0.0` (`reward: 0.0`) with an explicit reason in `report.json`. Setup, data, namespace and unexplained process failures exit without a reward.

## Separate replay from final scoring

Optional `--inference-only` replays both complete input releases and writes `inference-manifest.json`, without loading labels or fitting probes. `--score-existing` verifies SHA-256 hashes of model files, unlabeled inputs, feature files, replay/namespace/grader code and probe code, then performs frozen-probe scoring on CPU.

## Deployment checks

`docker-compose.yaml` builds from the task root and mounts `TASK_PRIVATE_DIR` read-only at `/data/eval`. `environment/Dockerfile` never copies `tests/` (except the label-free sandbox wrappers `tests/replay.py` and `tests/namespace.py`), keeping `tests/sealed/` and `tests/README.md` strictly isolated from `/workspace`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance
- **Upstream Benchmarks & Papers**: This task evaluates geospatial representation transfer across two complementary downstream regimes from **OlmoEarth** (*OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation*, Allen Institute for AI, arXiv:2511.13655; upstream commit `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`) and **Globe-LFMC 2.0** (*Yebra et al., Scientific Data 2024*):
  1. **African Wildlife Foundation (AWF) 9-class land-cover classification** (`allenai/olmoearth_projects_awf@da8eb6aab`), evaluated via the exact Table 2 non-parametric frozen **cosine-weighted `20-NN` classifier at softmax temperature $\tau = 0.07$** on unstandardized frozen embeddings (`olmoearth-paper-embeddings@6ea2c797`).
  2. **Live Fuel Moisture Content (LFMC) regression**, evaluated via frozen standardized $L_2$ ridge regression (`alpha` selected solely on validation) and scored by held-out **test Mean Absolute Error (MAE)**—explicitly MAE rather than RMSE, preserving robustness to extreme dry/wet canopy outliers in field destructive-sampling measurements and matching the primary skill calibration in `ssi_index_v3.py`.
- **Connected-Block Spatial Split Provenance**: Following spatial block cross-validation principles established in geospatial machine learning (*Rolf et al., Nature Communications 2021, "A generalizable and accessible approach to machine learning with global satellite imagery"*; *Roberts et al., Ecography 2017, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"*), the 1,459 AWF windows (which exhibit 1,546 overlapping raw window pairs) are grouped via connected-component union-find over intersecting 128-pixel UTM tiles (`EPSG:32737`) into 605 spatially disjoint blocks (`pairs_across_splits == 0`) and partitioned by `StratifiedGroupKFold` into `train = 1,053`, `valid = 201`, and blind `test = 205` (`roster_sha256: ed0cf3ce...`). All 205 blind `test` windows (including 156 formerly in upstream `train` and 49 in upstream `val`) are stripped from `/workspace/data/eval/awf/` prior to agent materialization.
- **Chance- & Constant-Calibrated V3/V4 Aggregation**:
  $$\text{AWF}_{\text{skill}} = \frac{\text{Acc}_{\text{test}} - 59/205}{1 - 59/205}, \qquad \text{LFMC}_{\text{skill}} = 1 - \frac{\text{MAE}_{\text{test}}}{24.435180349813233}, \qquad S = \operatorname{clip}_{[0,1]}\!\left(\frac{\text{AWF}_{\text{skill}} + \text{LFMC}_{\text{skill}}}{2}\right)$$
  Under constant embeddings, cosine `20-NN` predicts the training majority class and achieves `59/205` on the spatial test set ($\text{AWF}_{\text{skill}} = 0$), while predicting the training-label mean on LFMC yields $\text{MAE}_{\text{test}} = 24.435180349813233$ ($\text{LFMC}_{\text{skill}} = 0$), anchoring the trivial baseline at exact $S = 0.0$.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted
- **No End-to-End Fine-Tuning Gate or Custom Multi-Layer Probe**: OlmoEarth Table 2 specifically isolates representation quality of the pretrained frozen encoder via non-parametric `20-NN` classification and linear ridge probing. Replacing frozen probes with trainable non-linear decoders would confound backbone pretraining quality with downstream decoder engineering.
- **No Per-Component Zero Floor Prior to Averaging**: Individual signed skills (`awf_skill`, `lfmc_skill`) are averaged before top-level `[0, 1]` clipping (`ssi_index_v3.py:earth()`), preventing asymmetric reward gaming where catastrophic regression degradation (`MAE >> constant_MAE`) could be masked for free behind a `max(0, ...)` floor while over-optimizing a single modality.
- **No Artificial Runtime Penalty Gate**: Inference executes inside a network-isolated, label-free `bubblewrap` user namespace within the declared verifier budget (`task.toml`); imposing arbitrary latency multipliers would penalize legitimate architectural exploration (e.g., multi-scale temporal attention) without basis in the upstream remote-sensing literature.

### 3. Diagnostic Signals Retained in report.json
While `reward.json` exposes only flat numeric scalars (`score`, `reward`, `signed_score`, `awf_test_accuracy`, `awf_skill`, `lfmc_test_mae`, `lfmc_skill`, `awf_validation_accuracy`), `report.json` preserves rich scientific diagnostics for audit and analysis:
- Spatial `valid` (201 windows) and `test` (205 windows) `20-NN` hit counts, sample sizes, accuracies, and 9-class label distributions (`awf.valid`, `awf.test`).
- LFMC validation-selected ridge regularization `alpha`, validation/test **RMSE** and **$R^2$**, and empirical `constant_test` metrics (`lfmc.validation`, `lfmc.test`, `lfmc.constant_test`).
- Cryptographic provenance digests (`roster_sha256`, `awf_dev_splits_sha256`, `inference_manifest_sha256`, `label_file_sha256`) proving label-free replay and zero train/val-to-test window overlap.
