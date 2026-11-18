# Verification and frozen-package replay (V4)

Run commands from `brain-computer-interface-decoding/` with prepared `TASK_*_DIR`
paths outside Git. [Data acquisition and pinned releases](../environment/data/MATERIALIZE.md)
describe staging. [Baseline results](../environment/baseline/braindecode/RESULTS.md) and the
[acceptance record](../docs/acceptance-2026-10.md) hold historical reference replays.

## Active V4 4-Track Protocol & Security Controls

1. **Four Active EEG Tracks Only**: V4 evaluates and aggregates strictly the four subject/patient-disjoint EEG tracks (`tuev`, `ad65`, `openbmi`, `longitudinal`). The deprecated `FACED` track (confounded by recurring video stimuli and stimulus-locked EOG artifacts across subjects) is completely removed from active scoring and verification.
2. **Pretrained Weight De-Contamination**: `tests/main_verifier.py` enforces the V4 pretrained allowlist (`cbramod`, `codebrain`, or from-scratch `eegnet`/`contrawr`) and rejects `BIOT` (`braindecode/biot-pretrained-six-datasets-18chs`) by both metadata and SHA-256 (`a218f4054ae29962c6fd9d23d94c446d354c4f53e805a6771c85749ccc20baaa`) due to direct `TUEV` pretraining overlap.
3. **Per-Subject / Per-Recording-Session Isolated Inference**: `tests/main_verifier.py` partitions held-out test samples by `(subject, recording_session)` (parsing `(patient_id, session)` from `source` for `tuev` official evaluation windows), materializes a fresh single-unit shard (`shard-000000.npy`) containing only that unit's windows, runs `infer.sh` in an isolated temporary directory as `UID 10001`, kills lingering processes, and deletes unit artifacts before invoking the next unit. This physically prevents transductive feature whitening or BatchNorm adaptation across the test cohort.
4. **Exact V3/V4 Composite Score**: Both `reward.json["score"]` and `reward.json["reward"]` emit:
   $$\text{score} = \operatorname{clip}\!\left(\frac{1}{4}\sum_{k \in \{\text{tuev},\text{ad65},\text{openbmi},\text{longitudinal}\}} \frac{\text{BA}_k^{\text{subj/sess}} - c_k}{1 - c_k},\; 0,\; 1\right)$$
   with $c = \{\text{tuev}: \frac{1}{6},\; \text{ad65}: \frac{1}{3},\; \text{openbmi}: \frac{1}{2},\; \text{longitudinal}: \frac{1}{2}\}$, matching `harness/ssi_index_v3.py:bci()` within $< 10^{-12}$.

## Build and protocol checks

```bash
docker build -f environment/Dockerfile -t bci-development:0.4 .
docker build -f tests/Dockerfile -t bci-verifier:0.4 .
docker run --rm --network none -v "$PWD:/task:ro" -w /task \
  bci-development:0.4 \
  python -m unittest discover -s tests -p 'test_*.py' -v
python tests/container_probe.py --image bci-verifier:0.4
```

## Prepared-data and initialization checks

Only authorized operators run these checks with access to private evaluator assets:

```bash
docker run --rm --network none -v "$PWD:/task:ro" \
  -v "$TASK_DATA_DIR:/runtime/development:ro" \
  -v "$TASK_PRIVATE_DIR:/runtime/private:ro" \
  bci-development:0.4 \
  python /task/environment/data/validate_runtime.py /runtime
docker run --rm --network none -v "$PWD:/task:ro" \
  -v "$TASK_DATA_DIR/pretrained:/pretrained:ro" \
  bci-development:0.4 \
  python /task/environment/baseline/braindecode/validate_pretrained.py /pretrained
```

## Replay a frozen package

```bash
mkdir -p "$SCORE_DIR"
track_args=()
if [[ -n "${TRACK:-}" ]]; then track_args=(--tracks "$TRACK"); fi
docker run --rm --network none --cpus 4 --memory 20g \
  -v "$TASK_PRIVATE_DIR:/private/data:ro" \
  -v "$PACKAGE_DIR:/workspace/submission:ro" \
  -v "$SCORE_DIR:/logs/verifier" \
  bci-verifier:0.4 bash /tests/test.sh "${track_args[@]}"
```

Confirm `scope=full`, `inference_isolation=per_subject_session`, every requested track has status `ok`, and `reward.json["score"] == reward.json["reward"] == report.json["aggregate"]["score"]`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The V4 verifier evaluates cross-paradigm, subject-independent EEG decoding across four complementary clinical and BCI benchmarks following established EEG foundation-model evaluation protocols in **LaBraM** (*Jiang et al.*, "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI," *ICLR 2024*), **EEGPT** (*Wang et al.*, "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals," *NeurIPS 2024*), and **CBraMod** (*Wang et al.*, "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," *ICLR 2025*):

- **Removal of Confounded `FACED` & Overlapping `BIOT`**: Following V3/V4 scientific audit findings, `FACED` is excluded because all subjects view identical chronological video stimuli whose emotion labels are deterministic functions of clip identity (allowing stimulus-locked ocular/EOG artifacts to masquerade as affective EEG decoding), and the six-dataset `BIOT` checkpoint is excluded due to direct pretraining exposure to `TUEV`.
- **Subject/Session-Stratified Balanced Accuracy ($\text{BA}_k$)**: Following LaBraM, EEGPT, and CBraMod downstream evaluation conventions alongside clinical diagnostic hierarchy:
  - `tuev` ($K=6, c=1/6$): 6-class balanced accuracy across the official TUH Events patient-disjoint evaluation split (*Harati et al., IEEE SPMB 2015*);
  - `ad65` ($K=3, c=1/3$): patient-level balanced accuracy over 18 held-out patients after averaging window-level class probabilities per patient (preventing long recordings from dominating clinical diagnosis);
  - `openbmi` ($K=2, c=1/2$): per-subject motor-imagery balanced accuracy averaged equally across 11 held-out subjects (*Lee et al., GigaScience 2019*);
  - `longitudinal` ($K=2, c=1/2$): per-session balanced accuracy averaged within each subject across multi-day visits and then across 9 held-out subjects.
- **Chance-Normalized Equal-Weight Composite**: Because class cardinalities range from $K=2$ to $K=6$, raw balanced accuracies have unequal chance baselines ($1/6$ vs $1/2$). Linearly rescaling each track to Cohen/Youden-type chance-adjusted skill $( \text{BA}_k - 1/K ) / (1 - 1/K)$ sets random/constant guessing to $0.0$ and perfect decoding to $1.0$ on every track before taking the unweighted arithmetic mean and top-level $[0, 1]$ clip.
- **Per-Subject / Per-Recording-Session Isolated Inference**: In deployment, a clinical EEG reader or BCI decoder observes one patient visit or calibration-free session at a time, never an entire balanced multi-patient test cohort simultaneously. Running `infer.sh` on isolated `(subject, session)` shards preserves valid within-recording causality while blocking transductive cohort whitening and label-prior matching.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No Latency, FLOPs, or Parameter-Count Penalties in Primary Score**: LaBraM, EEGPT, and CBraMod benchmark representation transfer strictly via held-out classification accuracy/balanced accuracy under fixed wall-clock execution limits (`timeout = 7200s` per track). Penalizing parameter count would artificially favor tiny single-paradigm CNNs (e.g., 9K-parameter EEGNet) over multi-task foundation backbones.
- **No Weighted-F1 or Cohen's $\kappa$ Mixing in the Scalar Composite**: While LaBraM and CBraMod report balanced accuracy, Cohen's $\kappa$, and weighted F1 on multiclass TUEV, mixing heterogeneous metric families (F1 on one track, BA on another) distorts cross-track scale invariance; chance-adjusted balanced accuracy provides a single mathematically uniform linear skill measure across binary and multiclass tracks.

### 3. Diagnostic Signals Retained in report.json

`report.json` retains comprehensive clinical and statistical diagnostics outside the primary scalar reward:
- **Window-Pooled vs Subject/Session-Stratified Metrics**: Both `primary_balanced_accuracy` (used in `reward.json`) and `window_balanced_accuracy`, `subject_metrics` (`ad65`), and per-subject `subject_scores` (`openbmi`, `longitudinal`).
- **Multiclass & Agreement Diagnostics**: Per-track `confusion_matrix`, `per_class_recall`, `class_counts`, `predicted_class_counts`, `accuracy`, `weighted_f1`, `macro_f1`, and `kappa` (Cohen's $\kappa$, standard secondary metric in LaBraM/CBraMod/EEGPT on TUEV).
- **Audit & Isolation Provenance**: `inference_isolation` (`"per_subject_session"`), `isolated_inference_units`, `sample_count`, `subject_count`, `seconds`, `data_identity`, `package_sha256`, and unclipped `aggregate.signed_score` / per-track `aggregate.terms`.
