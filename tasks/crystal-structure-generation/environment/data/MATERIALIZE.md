# Materializing CrystalGen data (operators)

Controller-side. The agent never runs these scripts; they are not copied into the image. The result
is one host directory used as both `TASK_DATA_DIR` and `TASK_PRIVATE_DIR`:

```text
$ROOT/
  mp20/train.csv, val.csv, test.csv, manifest.json    development splits; test.csv stays on the host
  scorer-assets/                                      prepared scorer inputs (about 52 GB), READY.json
  final-suite/requests.jsonl, suite_manifest.json     the verifier's final request (private)
```

The development container mounts `mp20/train.csv`, `mp20/val.csv`, `mp20/manifest.json` and
`scorer-assets/` read-only under `/workspace/data`. The verifier's replay container mounts only
`final-suite/`; its grader mounts `final-suite/` and `scorer-assets/`.

All commands run in the task's development image, with this directory mounted, from the repository
root:

```bash
docker build -t crystalgen-dev -f crystal-structure-generation/environment/Dockerfile crystal-structure-generation
export ROOT=/secure/crystalgen
TASKDATA="-v $PWD/crystal-structure-generation/environment/data:/task-data:ro -v $ROOT:$ROOT"
```

## 1. MP-20

Public; no approval needed. The script fetches `txie-93/cdvae` at
`f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea`, parses the CSVs with Python's CSV parser (CIF fields
are multiline), adds `density` from each CIF with pymatgen, and validates the measured counts
(27,136 / 9,047 / 9,046) and SHA-256 values:

```bash
docker run --rm $TASKDATA crystalgen-dev bash /task-data/materialize.sh "$ROOT/mp20"
```

## 2. Scorer assets

CrystalGen uses the exact three-potential LeMat-GenBench evaluation during development and final
grading. The pre-scoring relaxation reuses the same UMA checkpoint and cache.

| Asset | Revision/model identifier | License position |
| --- | --- | --- |
| LeMat-Bulk `compatible_pbe` | `0dc17eea904b860ad7288141e9870f67f8e6bb2c` | CC BY 4.0 |
| LeMat-Bulk MLIP Hull | `70d505bb294c16658a9551e2b81ace69a28d9790` | No license tag; review required |
| UMA | `facebook/UMA` revision `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2`, model `uma-s-1`, task `omat` | Manual access; FAIR Chemistry License |
| ORB | `orb_v3_conservative_inf_omat` | Apache-2.0 code/model family |
| MACE | `mace_mp()` from `mace-torch==0.3.13` | MIT code |

ORB and MACE resolve model files indirectly; `READY.json` records the resulting cache-tree digest,
and the grader refuses assets whose digest changed.

Approval gates:

- A person must request and receive access to `facebook/UMA` on Hugging Face (FAIR Chemistry
  License and identity disclosure). The task cannot accept it on that person's behalf. The pinned
  protocol uses archived `uma-s-1`; its model card reports a known extensivity bug. Changing the
  checkpoint creates a new, non-comparable evaluation version.
- `LeMaterial/LeMat-Bulk-MLIP-Hull` exposes no license metadata. The operator must approve its use
  under their institution's policy or obtain clarification before passing
  `--acknowledge-hull-license-review`. That flag records a decision; it is not a license grant.

Prepare on a GPU host after `hf auth login`. The token file is visible to this one container,
read-only, and is not copied into the image or the asset tree (use the path printed by `hf env`):

```bash
docker run --rm --gpus all $TASKDATA \
  -e HF_HUB_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v /absolute/path/to/huggingface/token:/run/secrets/hf-token:ro \
  crystalgen-dev \
  bash /task-data/scorer_assets/materialize.sh "$ROOT/scorer-assets" \
    --acknowledge-hull-license-review --warm-mlips --token-file /run/secrets/hf-token
```

Resulting layout:

```text
huggingface/hub/      pinned hub snapshots and audited main -> commit refs
huggingface/datasets/ prepared datasets cache used only during validation
home/.cache/          persisted ORB, MACE and FairChem model caches
xdg-cache/            library caches that honor XDG_CACHE_HOME
torch-cache/          torch model cache, if used upstream
READY.json            pinned revisions and a content-tree digest
```

`materialize.sh` ends with `validate.py`; preparation fails rather than silently omitting UMA or a
hull dataset. Keep the tree read-only afterwards: runtime writes change the digest.

## 3. Final request suite (private)

From an operator-owned seed file with at least 32 random bytes; the builder refuses to overwrite an
existing suite:

```bash
python3 crystal-structure-generation/tests/requests/build_suite.py \
  --seed-file "$ROOT/private/request-seed" --output-root "$ROOT/final-suite"
```

It writes one 2,500-sample unconditional request and a `crystalgen-final-suite/v1` manifest. Only
the seed is private; there is no answer key. See `tests/requests/README.md`.
