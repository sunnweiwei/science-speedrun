# Crystal Structure Generation

## 1. Overview

CrystalGen asks an agent to train one portable generator on the standard MP-20 training split and
keep improving it. The task is **unconditional generation only**: generate 2,500 fresh structures
and maximize SUN + MetaSUN with the pinned LeMat-GenBench scorer. There are no requested chemical
systems, density targets or conditional rewards.

This matches the scope of the [public LeMat-GenBench benchmark](https://huggingface.co/spaces/LeMaterial/LeMat-GenBench)
and its [unconditional evaluation protocol](https://arxiv.org/html/2512.04562v1#S3.SS1).
The grader first relaxes a separate copy of each submitted crystal with the frozen UMA
`uma-s-1` (`omat`), BFGSLineSearch and FrechetCellFilter, up to 200 steps at 0.02 eV/Å and
300 seconds per crystal. It retains finite unconverged iterates and uses the original
candidate when refinement fails. It then scores the resulting geometry with ORB-v3, MACE-MP
and UMA. Mean energy above hull defines stable (≤0 eV/atom) and metastable (0–0.1 eV/atom)
bins, with stable structures excluded from the metastable bin. The reward is the fraction of all
requested slots that are unique within a bin and novel relative to LeMat-Bulk.

For equal compositions, CrystalGen applies MatterGen's v1.0.3 ordered-disordered matcher.
The additional dilute-parent guard identifies a known lattice decorated with a substituent
occupying ≤25% of the host-plus-guest sites. It requires relative radius difference ≤0.3 and
electronegativity difference ≤1.0. Supercells, symmetry copies and dilute variants of a known
compound such as GdDy7Cu8 from DyCu are covered by these comparisons. A cross-element substitution requires both the measured radii and the electronegativities. Ordinary same-species geometric matches still apply to materials such as XeF₂ when a radius is unavailable.

The optional `--as-submitted` evaluation scores the raw lattice and coordinates. It also
measures displacement from the raw geometry under all three potentials on 250 prespecified
slots (50 steps, 0.02 eV/Å), with coverage and a 95% structure-bootstrap interval.
This result is recorded separately from the primary UMA-200 reward.

The generation package runs offline on the sealed request and has no access to scoring assets.

The MACE energy and RMSD checkpoint is `medium-0b3`; the MACE distribution embedding checkpoint
is `medium-mpa-0`. The former matches the energy scale of `mace_mp_energy` in the published
reference hull. ORB uses `orb_v3_conservative_inf_omat` and UMA uses `uma-s-1` with its `omat` head.
The upstream authors document a small unresolved discrepancy for the UMA reference energies.
[LeMat checkpoint correction](https://github.com/LeMaterial/lemat-genbench/blob/fbf1ba4855934acb8fe87135315504899064080d/README.md) · [LeMat Appendix S2.1](https://arxiv.org/html/2512.04562v3)

For an existing validated asset tree, the operator may hardlink a separate revision on the same
filesystem with `environment/data/scorer_assets/upgrade_mace.py OLD_ASSETS NEW_ASSETS CHECKPOINT`.
Use the [official MACE-MP-0b3-medium file](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model).
The helper verifies the prior manifest and the new release hash and preserves the v1 tree.
Fresh materializations warm and check both MACE models.

## 2. Data

| Source | Pinned revision | Size | License |
| --- | --- | --- | --- |
| MP-20 (`data/mp_20` in `txie-93/cdvae`) | `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea` | < 0.5 GB | Materials Project data, CC BY 4.0; CDVAE code MIT |
| LeMat-Bulk `compatible_pbe` | `0dc17eea904b860ad7288141e9870f67f8e6bb2c` | part of 52.5 GB assets | CC BY 4.0 |
| LeMat-Bulk MLIP hull | `70d505bb294c16658a9551e2b81ace69a28d9790` | | no license tag; operator review |
| UMA `uma-s-1` (`omat`) | `facebook/UMA` `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2` | | FAIR Chemistry License, gated |
| ORB `orb_v3_conservative_inf_omat`, MACE `medium-0b3` (energy), `medium-mpa-0` (embedding) (`mace-torch==0.3.13`) | cache digest in `READY.json` | | Apache-2.0 / MIT |

The task validates the measured MP-20 counts (27,136 / 9,047 / 9,046 = 45,229) and source-file
SHA-256 values rather than the stale 45,231-row claim in the upstream README.

[`environment/data/MATERIALIZE.md`](environment/data/MATERIALIZE.md) has the complete procedure. In
short, from the repository root, with the development image built:

```bash
export ROOT=/secure/crystalgen
TASKDATA="-v $PWD/crystal-structure-generation/environment/data:/task-data:ro -v $ROOT:$ROOT"
docker run --rm $TASKDATA crystalgen-dev bash /task-data/materialize.sh "$ROOT/mp20"
docker run --rm --gpus all $TASKDATA \
  -e HF_HUB_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v /absolute/path/to/huggingface/token:/run/secrets/hf-token:ro crystalgen-dev \
  bash /task-data/scorer_assets/materialize.sh "$ROOT/scorer-assets" \
    --acknowledge-hull-license-review --warm-mlips --token-file /run/secrets/hf-token
python3 crystal-structure-generation/tests/requests/build_suite.py \
  --seed-file "$ROOT/private/request-seed" --output-root "$ROOT/final-suite"
```

```text
$ROOT/mp20/{train,val,test}.csv, manifest.json   test.csv is never mounted
$ROOT/scorer-assets/                             52.5 GB, READY.json with the tree digest
$ROOT/final-suite/                               one 2,500-sample request (private seed)
```

Operator approval gates:

- No approval is needed for MP-20. Its test split stays on the host.
- A person must accept the `facebook/UMA` FAIR Chemistry License and identity disclosure on
  Hugging Face. The task cannot accept it on that person's behalf.
- The pinned leaderboard protocol uses archived UMA `uma-s-1`. Its model card reports a known
  extensivity bug and recommends newer checkpoints. Keeping `uma-s-1` preserves comparison with
  the pinned protocol; changing it creates a new, non-comparable evaluation version.
- `LeMaterial/LeMat-Bulk-MLIP-Hull` exposes no license metadata. The operator must approve its use
  or obtain clarification before passing `--acknowledge-hull-license-review`. That flag records a
  decision; it is not a license grant.

## 3. Workspace

```text
/workspace/
  instruction.md, contract.md
  data/README.md
  data/mp20/{train.csv,val.csv,manifest.json}   read-only mounts
  data/scorer-assets/                           read-only mount (also /scorer-assets)
  baseline/starter/                             OMatG starter + smoke generator, README, RESULTS
  references/README.md                          OMatG/LeMat sources, historical results, papers
  evaluation/                                   score.sh, relax.sh, score.py, relax.py, examples/
  submission/                                   empty at start
  research/                                     empty, writable, large disk
/opt/omg, /opt/lemat, /opt/crystalgen/scoring   OMatG source, pinned scorer, shared scoring code
```

### Data

| Resource | Visible? | Permitted use |
| --- | --- | --- |
| MP-20 train: 27,136 rows | Yes | The only structures permitted for fitting model parameters |
| MP-20 validation: 9,047 rows | Yes | Model selection and diagnostics; do not fit on it |
| MP-20 test: 9,046 rows | No | Not mounted in development; not needed to build the unconditional final suite |
| Per-row labels | Yes | Optional training-split analysis: formation energy, band gap, energy above hull, space group, and density derived from each CIF; no target conditions required |
| LeMat-Bulk reference structures | Yes, read-only | Development novelty evaluation only; never candidate training or retrieval |
| MLIP hull tables | Yes, read-only | Development stability evaluation only |

### Model starting point

| Resource | Visible? | Rule |
| --- | --- | --- |
| OMatG source at `ed80cb86...` | Yes | Recommended research starting point |
| Unconditional OMatG starter | Yes | Upstream model and encoder without property adapters; offline 8-GPU replay |
| Published OMatG or other generator weights | No | Forbidden as initialization, distillation targets, or package contents |
| ORB, MACE, and restricted UMA scoring weights | Yes, read-only | May run the exact final scorer; must not initialize, train, distill, or be copied into the generator |

The starter has 12,403,912 learned parameters and uses upstream `Model`, `CSPNetFull` with
`prop: false`, and `StructureDataset` without property keys. All training examples teach
unconditional generation: there are no condition adapters, condition dropout or request-driven
species/density projections. A small CLI compatibility shim accepts the linked batch-size argument
without changing base-state sampling. The Linear-SDE-Gamma recipe is an editable starting point; it
does not claim to reproduce the public OMatG training recipe. A separate statistical generator is
included only for fast interface smoke tests. Submissions hold one parameter set trained from
random initialization during the task, with at most 70 million trained parameters.

OMatG is the trained baseline; the statistical generator is a separate control. Results under the archived UMA-200 protocol are listed in section 6.

The shared generator image applies `omatg-generator-runtime.patch` to OMatG's packaging and import
wiring. The pinned OMatG release supports Python 3.11, but its optional energy path declares
TorchSim 0.5 (Python 3.12 only) as mandatory and imports it even during ordinary fitting. The patch
removes that optional dependency and defers its import until the unused energy-analysis command is
called. No model architecture, training math or sampling code is changed.

### Development evaluation

Run `/workspace/evaluation/score.sh INPUT GENERATED SCORE_DIR` to perform the fixed UMA-200 relaxation and score its output. See the [development guide](environment/evaluation/README.md).

## 4. Evaluation

### Untrusted model replay

The replay container (`main` in [`tests/docker-compose.yaml`](tests/docker-compose.yaml)) receives
only:

- a root-owned, read-only snapshot of the validated `/workspace/submission` package;
- input-only unconditional request files;
- an empty writable output directory.

It does **not** receive MP-20, final-scoring-ensemble weights, hull data, LeMat-Bulk, credentials,
a scoring endpoint or network access. It runs `bash generate.sh INPUT_DIR OUTPUT_DIR` from the
snapshot as UID 10001, validates every expected row, kills remaining submission processes, copies
the output into a read-only hash-bound snapshot, and only then signals the grader. A package that
fails validation, exits nonzero or exceeds the 24-hour inference limit gets
`{"reward": 0.0, "invalid_submission": ...}` and nothing is scored.

### Trusted grading

The grader receives the sealed output, the requests, the pinned scorer and the prepared assets. It
never imports or executes submission code. The grader records hashes of the raw and relaxed CIFs and stages the latter in stable request/sample order. `reward.json` holds the SUN+MetaSUN fraction; `report.json` records per-structure diagnostics. The primary schema is `crystalgen-report/v4-uma200-hull-aligned`.

### Metrics

- `score` (= `reward` = `unconditional_ssi_score`): `(SUN + 0.5 * MetaSUN) / 2,500`, the equal mean of strict SUN yield `a = SUN / 2,500` and stable-or-metastable SUN yield `b = (SUN + MetaSUN) / 2,500`.
- `unconditional_sun_metasun`: fraction of all 2,500 requested structures that are
  either SUN or MetaSUN (`SUN + MetaSUN` yield, retained for upstream leaderboard comparison).
- `unconditional_sun`: fraction passing strict stability, uniqueness and novelty.
- `unconditional_metasun`: fraction passing metastability (excluding the strict stable bucket),
  uniqueness and novelty.
- `unconditional_validity`: valid structures divided by all requested outputs.
- `unconditional_uniqueness` and `unconditional_novelty`: unique and novel structures divided by all
  requested outputs.

Missing, duplicate, malformed, unknown or unparsable outputs stay in the denominator and count as
failures. `report.json` also preserves the upstream leaderboard diagnostics: stable and metastable
counts; novelty and uniqueness; formation energy and energy above hull; ensemble uncertainty and
relaxation RMSD; element, space-group, site-count and physical-size diversity; JSD, MMD and Fréchet
distribution distances; and production/reserve HHI. None is folded into the reward.

## 5. Running

```bash
export TASK_DATA_DIR=/secure/crystalgen TASK_PRIVATE_DIR=/secure/crystalgen   # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/crystalgen-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p crystal-structure-generation -a <agent> -m <model> --allow-agent-host <model API host>
```

**Resources (external budget, in `task.toml`).** The agent container has eight H100s, 64 CPU cores,
512 GiB of memory and a 48-hour agent timeout; the verifier also uses eight H100s. For scale: the
2,000-epoch starter took 5.4 h on six H100s. Under the archived UMA-200 protocol, scoring 2,500 structures took about 1.2 h of
relaxation plus 1.5–1.7 h of LeMat scoring on eight H100s with one relaxation worker per GPU; with
several CPU-bound workers per GPU (the default now, 64 verifier CPUs) a full verifier run took 2 h 9 min
end to end on four RTX 6000 Ada: 27 min relaxation, 61 min multi-MLIP preprocessing, 14 min novelty,
13 min uniqueness and ~10 min SUN/stability.

Smoke ladder, cheapest first:

1. **Contract** (no data, Docker or GPU): `python3 -m unittest discover -s crystal-structure-generation/tests -p 'test_*.py' -v`.
   Request and package schemas, fixed denominators, hidden-suite shape, scorer-copy identity,
   invalid-package scoring and final generation/grading isolation. Not a scientific test.
2. **Container and public data** (no scorer assets needed for the generator):

   ```bash
   docker compose -f crystal-structure-generation/environment/docker-compose.yaml up -d --build
   docker compose -f crystal-structure-generation/environment/docker-compose.yaml exec main bash -lc '
     bash /workspace/baseline/starter/train.sh /workspace/data/mp20/train.csv /workspace/research/smoke/submission
     bash /workspace/research/smoke/submission/generate.sh /workspace/baseline/starter/examples/input /workspace/research/smoke/output
     python3 -c "
   from pathlib import Path
   from scoring.contract import load_output, load_requests, validate_submission
   validate_submission(Path(\"/workspace/research/smoke/submission\"))
   records, errors = load_output(Path(\"/workspace/research/smoke/output\"), load_requests(Path(\"/workspace/baseline/starter/examples/input\")))
   assert len(records) == 2 and not errors, (len(records), errors)
   print(\"PASS\")"'
   ```

3. **Exact scorer** (prepared assets, eight GPUs): score the 100-structure example batch with
   `bash /workspace/evaluation/score.sh /workspace/evaluation/examples/input OUTPUT SCORE_DIR`.
   Success means all GPUs are visible, all three potentials load offline, both hull/reference
   datasets resolve, the LeMat pipeline exits zero and `reward.json` is finite.
4. **Verifier**: `docker compose -f crystal-structure-generation/tests/docker-compose.yaml up` with a package bound at
   `/workspace/submission` of `main`, then `bash /tests/test.sh` in `main`.

## 6. Baseline results and validation

The corrected v4 primary yielded **0.1752** for the released 2,000-epoch starter on its original 2,500 generated CIFs and fixed UMA-200 endpoints, with 8 SUN and 430 MetaSUN. All 2,388 valid candidates had three finite hull energies and a novelty decision. [Baseline measurement](environment/baseline/starter/reproduction/v4_score_summary.json)

### Earlier scorer versions

Reproduced by the task reviewers on one 8×H100 host (details in
[`environment/baseline/starter/RESULTS.md`](environment/baseline/starter/RESULTS.md)); 2,500
unconditional structures, fixed UMA-200 relaxation, unchanged LeMat scoring:

| generator | SUN + MetaSUN | SUN | MetaSUN | validity | uniqueness | novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke generator (no learning) | 0.106 | 0.0088 | 0.0972 | 0.8724 | 0.7864 | 0.6748 |
| OMatG starter, 2,000 epochs from scratch | **0.208** | 0.0060 | 0.2020 | 0.9552 | 0.9504 | 0.6852 |

The task author's historical v0 checkpoints (conditional adapters, trained on Modal) reached 21.04 %
(Linear-SDE-Gamma) and 19.56 % (EncDec-ODE-Gamma) on the same pipeline; before relaxation they
scored 1.88 % and 2.64 %. The historical results are listed as references, not as measurements of
the current starter. Author-side acceptance (unit tests, Modal end-to-end smoke, scorer parity with
one and eight workers, the asset-tree audit) is recorded in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## 7. Limitations

- Under the archived UMA-200 protocol, about half of the starter's reward was reached by the no-learning smoke generator: the fixed UMA
  relaxation moves many plausible compositions onto the metastable side of the hull. Strict SUN is
  below 1 % for both.
- Evaluation is expensive: about two hours for one 2,500-structure score (2 h 9 min measured on
  four RTX 6000 Ada with 64 CPUs; novelty, uniqueness and SUN matching are single-threaded upstream).
- The full fresh-seed replay of a 2,500-structure final suite through the Harbor verifier on a
  Docker host with the NVIDIA runtime and `network_mode: none` has not been run end to end; the
  pieces (generation, sealing, relaxation, scoring) were each run on the same host.
- UMA `uma-s-1` has a known extensivity bug; it is kept for comparability with the pinned protocol.
- The MLIP hull dataset has no license tag; each operator must make its own decision.
- The final seed is private, but the request schema, scorer and assets are public, so this is a
  benchmark-visible evaluation rather than a secret test set.

## 8. Citation and licenses

Task contributor: [junlin-star](https://github.com/junlin-star).

- OMatG: P. Hoellmer et al., *Open Materials Generation with Stochastic Interpolants*, ICML 2025
  (arXiv:2502.02582); MIT.
- LeMat-GenBench (arXiv:2512.04562) and LeMat-Bulk (CC BY 4.0).
- MP-20: T. Xie et al., *Crystal Diffusion Variational Autoencoder for Periodic Material
  Generation*, ICLR 2022; Materials Project data under CC BY 4.0, CDVAE code MIT.
- UMA (FAIR Chemistry License), ORB (Apache-2.0), MACE (MIT).

The data and weights are not redistributed in this repository.
