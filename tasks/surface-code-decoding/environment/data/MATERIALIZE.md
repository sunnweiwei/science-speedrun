# Immutable materialized data release

The task uses the Google Quantum AI Sycamore surface-code memory experiments,
[Zenodo 6804040](https://zenodo.org/records/6804040), CC-BY-4.0. It includes four
distance-3 patches and one distance-5 patch, both X/Z bases, with 13 odd durations
1..25: 130 conditions and 6.5 million shots. Per condition, training takes the
first 19,880 even rows, validation the last 5,120 even rows, and test all 25,000
odd rows (zero-based source indexing). Arrays no longer expose source row indices.

The versioned HF release has three separate archives:

| Archive | Extracted root | Access / purpose |
|---|---|---|
| `development.tar.gz` | `development/` | Public; 2,584,400 training shots, 665,600 validation shots, allowed circuits/DEM, and `reference/`: the reference model's results and a weightless submission template (no trained weights) |
| `verifier-inputs.tar.gz` | `verifier/` | Public historical inputs; 3,250,000 test shots, opaque IDs, detection events and ideal circuits only; never mounted in agent development |
| `grader-labels.tar.gz` | `grader/` | Private evaluator distribution; IDs and test labels, accessible to trusted grader only |

Public archives are in `sunweiwei/ai4sci-surface-code-decoding`; grader labels, and
the frozen reference model operators use for known-answer replays
(`reference-model.tar.gz`, never mounted for agents), are in the private
`sunweiwei/ai4sci-surface-code-decoding-evaluation`. Release v1 was first published
under `Corning/`; v1.1 changes only the development archive (no weights) and moves
all archives to the repositories above, with identical verifier/grader bytes. The public upstream already
contains the historical test, so this distribution separation is an execution
boundary, not a claim of test secrecy.

`hf_release.json` pins every file to a full HF commit SHA, size and SHA-256.
Downloads use `hf_hub_download`'s local cache and recovery. No release follows
HF `main`. Payload inventories provide exact coverage and per-file integrity;
unsafe tar paths, links and oversized expanded archives are rejected.

```bash
# Only the operator needs network or HF credentials, before launching agents.
uv run --no-project --with huggingface-hub==0.35.3 \
  python environment/data/materialize.py --output /absolute/path/qec-data \
  --roles development verifier

# Requires authorized HF login/token; never propagate it into agent containers.
uv run --no-project --with huggingface-hub==0.35.3 \
  python environment/data/materialize.py --output /absolute/path/qec-data \
  --roles grader

export TASK_DATA_DIR=/absolute/path/qec-data
export TASK_PRIVATE_DIR=/absolute/path/qec-data/grader
```

An offline mirror may pass `--archive-dir PATH`; all pinned archive hashes are
still enforced. Repeating materialization validates existing role contents and
refuses a different release or corruption rather than silently replacing data.
On an empty machine, allow approximately 1.1 GB for extracted data plus archives,
plus the pinned GPU runtime and working model space. Scientific source generation
is not needed for normal deployment.

The official score is computed by trusted Git/Docker code, not any scorer
shipped inside a submitted model or the development release; verifier hardening
can be updated in Git without changing the immutable data release.

Without access to the private repository, the verifier and grader archives can be
rebuilt from the public Zenodo source with `build_release.py`: the integrator
rebuilt both and matched the pinned SHA-256 byte for byte. Pass the rebuilt
archives with `--archive-dir`.

## Rebuild and audit

`build_release.py` uses the exact original source archive and the reference
results written by `make_reference_results.py` (which replays the frozen reference
model on validation; its weights stay operator-side). It writes deterministic archives (stable order,
fixed timestamps/owner/modes, deterministic NPZ contents), with train/validation
labels physically separated from test. Noise resources are allowlisted to ideal
circuits and `pij_from_even_for_odd.dem`; the latter is calibrated using all even
syndromes, including validation syndromes, but no logical labels or odd shots.
No historical decoder predictions or full raw data are copied into development.

```bash
# Run with the pinned AlphaQubit runtime (NumPy + Stim installed).
python environment/data/materialize.py --from-sources \
  --source /path/to/original/sycamore \
  --reference-results /path/to/reference-results \
  --output /path/to/new-release
```

The builder returns `materialized/{development,verifier,grader}`, the three
archives and `release-manifest.json`. Its archive bytes are deterministic for
the exact source data, reference results, adapter sources and dependency
versions used. Later adapter/verifier changes can change development archive
bytes; the published lock remains the authority for the frozen release. Updating assets requires an intentional
new HF revision and reviewed lock changes. Never publish the private grader
archive in the public repository, and never mount the whole materialization
root into the development container.
