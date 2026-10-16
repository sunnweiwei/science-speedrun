# One Hugging Face bundle

The public [dataset](https://huggingface.co/datasets/junlinw/ai4sci-atomistic-materials)
contains the 500k training / 5k labeled-dev sample, four model checkpoints, prepared
physical validation/test panels, frozen split metadata and both trials' results.
Git contains code, documentation, release pins and figures; no data or weights.

```bash
# Operator host: prepare training, initialization and physical validation.
uv run --frozen python environment/data/materialize.py \
  --components training protocol initial-model eval-val
# Download the reference checkpoints and measured comparison.
uv run --frozen python environment/data/materialize.py \
  --components trained-models records
# Final evaluator only: prepare test data without running model evaluation.
uv run --frozen python environment/data/materialize.py --components eval-test
```

With no `--components`, only `training` and `protocol` are selected. License notices
are always included. Models, validation and test are opt-in; selecting validation
never downloads test files. The bundle is public and does not need HF credentials.
UMA files and derivatives retain the FAIR Chemistry License and Acceptable Use
Policy; data retain their component licenses. See the HF `LICENSES.md`.

| Component | Materialized path under `.data/` | Role |
| --- | --- | --- |
| `training` | `standard-v1/data/` | 500k fitting / 5k labeled-dev frames, manifest and elemental references |
| `protocol` | `splits/v1/`, `panels/v1/`, `evaluation-sources.json` | Frozen membership and source provenance |
| `initial-model` | `models/uma-s-1p1.pt`, source metadata | Original checkpoint used to initialize both trials |
| `trained-models` | `models/{3ep,30ep-selected,30ep-final}/`, `models/index.json` | Exported reference models with energy audits and manifests |
| `eval-val` | `physical-v1-val/{inputs,references}/` | 1,200 physical validation cases |
| `eval-test` | `physical-v1-test/{inputs,references}/` | 6,000 cases for reserved final grading |
| `eval-regression` | `physical-v1-regression/` | 28 diagnostic cases |
| `records` | `records/` | Both trials' curves, physical predictions, scores and timing audit |
| `licenses` | `licenses/` | Automatically included upstream terms and attribution |

The downloader pins an immutable HF commit, checks the manifest SHA-256, verifies
compressed and unpacked sizes/hashes, and atomically publishes complete files.
Repeated runs verify existing files; corrupt files are rejected. `manifest.json`
provides exact sizes per component. The complete bundle is about 3.43 GB to download;
training plus protocol is about 479 MB. Disk usage also includes the compressed cache.
The four checkpoint files are approximately 1.17 GB + 3 × 587 MB.

Physical inputs and references are byte-identical to the already-prepared panels.
Keep test references and the operator source cache outside development/prediction
mounts; only the trusted scorer receives targets. Upstream targets are public, so
this is runtime separation rather than secret source data. Packaging the test panel
does not evaluate it; the 6,000-case test remains unused.

The original UMA checkpoint uses upstream energy conventions; the three adapted
checkpoints use MP-compatible raw total energy and their adjacent energy audits.
MPTrj training uses `uncorrected_total_energy` in eV and forces in eV/Å; VASP kbar
stress is converted to tensile-positive ASE eV/Å³. Released sAlex units are retained.
No MP2020 correction is applied during fitting. Published MPA elemental references
are allowed pretrained resources; force RMS is fitted on training only.

## Rebuild from upstream sources

The unchanged training files preserve the original 20.3 GB source-sampling result.
To reproduce sampling, materialize `protocol`, then:

```bash
uv run --frozen --extra data python environment/data/prepare_standard.py \
  --source mptrj --root .data/rebuilt --cache .data/full-training-sources
uv run --frozen --extra data python environment/data/prepare_standard.py \
  --source salex --root .data/rebuilt --cache .data/full-training-sources
uv run --frozen --extra data python environment/data/prepare_standard.py \
  --source assemble --root .data/rebuilt --cache .data/full-training-sources
# Optional independent reconstruction of the frozen physical validation panel.
uv run --frozen --extra data python environment/evaluation/prepare.py \
  --preset val --panel .data/panels/v1/manifest.json \
  --output .data/rebuilt-physical-val --workers 8
```

Training manifests preserve source hashes, URLs, units and overlap checks. MPTrj
comes from [Figshare v2](https://doi.org/10.6084/m9.figshare.23713842.v2) (MIT);
sAlex and elemental references from [OMat24](https://huggingface.co/datasets/facebook/OMAT24)
(CC BY 4.0). Physical source pins remain in `../evaluation/sources.json`. Source
reconstruction downloads about 163 MB of common files plus selected phonon YAMLs.
The original gated UMA source downloader remains available for independent checks.
