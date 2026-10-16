# Crystal Structure Generation

## 1. Overview

CrystalGen asks an agent to train one portable generator on the standard MP-20 training split and
keep improving it. The task is **unconditional generation only**: generate 2,500 fresh structures
and maximize SUN + MetaSUN with the pinned LeMat-GenBench scorer. There are no requested chemical
systems, density targets or conditional rewards.

This matches the scope of the [public LeMat-GenBench benchmark](https://huggingface.co/spaces/LeMaterial/LeMat-GenBench)
and its [unconditional evaluation protocol](https://arxiv.org/html/2512.04562v1#S3.SS1).
Scoring-protocol compatibility does not imply identical training recipes or generation-time
postprocessing to a leaderboard submission. The task evaluates **generator + fixed UMA-200
postprocessing**, not raw-generator quality alone; the public benchmark permits already-relaxed
submitted structures (see its [pre/post-relaxation discussion](https://arxiv.org/html/2512.04562v1#S5.SS3)).

The agent uses the exact final evaluation stack during development, which gives it useful research
feedback. Final replay is still isolated: the submission generates structures without network or
scorer access, its output is sealed, and a trusted grader relaxes a copy before scoring it.

- **Reward:** `unconditional_sun_metasun`, the fraction of the 2,500 requested structures that are
  stable-unique-novel or metastable-unique-novel after the fixed relaxation (section 4).
- **Status:** verified. The OMatG starter trained from scratch reaches **0.208**; the no-learning
  smoke generator reaches 0.106 (section 6).

Terminology:

- **Final scoring ensemble**: the fixed ORB + MACE + UMA evaluator. It scores candidates; it is not
  part of the submitted generator.
- **Restricted scoring checkpoint**: UMA `uma-s-1`, whose files require an approved Hugging Face
  account. "Restricted" describes download access, not a different scientific role.
- **Pre-scoring relaxation**: optimizing atom positions and the cell with frozen UMA weights, not
  training a neural network. It is a fixed grader step: BFGSLineSearch + FrechetCellFilter,
  `fmax=0.02` eV/Å, at most 200 steps and 300 seconds per structure. Development uses the identical
  helper.
- There is no "oracle" or query allowance: during development the agent can run the exact final
  scorer as often as its compute budget permits; during final generation it cannot access it.

## 2. Data

| Source | Pinned revision | Size | License |
| --- | --- | --- | --- |
| MP-20 (`data/mp_20` in `txie-93/cdvae`) | `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea` | < 0.5 GB | Materials Project data, CC BY 4.0; CDVAE code MIT |
| LeMat-Bulk `compatible_pbe` | `0dc17eea904b860ad7288141e9870f67f8e6bb2c` | part of 52.5 GB assets | CC BY 4.0 |
| LeMat-Bulk MLIP hull | `70d505bb294c16658a9551e2b81ace69a28d9790` | | no license tag; operator review |
| UMA `uma-s-1` (`omat`) | `facebook/UMA` `f611b917d9c68566bbbeccbb0aa0f7cad1696cb2` | | FAIR Chemistry License, gated |
| ORB `orb_v3_conservative_inf_omat`, MACE `mace_mp()` (`mace-torch==0.3.13`) | cache digest in `READY.json` | | Apache-2.0 / MIT |

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

| baseline | reproduced SUN + MetaSUN |
| --- | ---: |
| smoke generator (no learning) | 0.106 |
| OMatG starter, 2,000 epochs | 0.208 |

The shared generator image applies `omatg-generator-runtime.patch` to OMatG's packaging and import
wiring. The pinned OMatG release supports Python 3.11, but its optional energy path declares
TorchSim 0.5 (Python 3.12 only) as mandatory and imports it even during ordinary fitting. The patch
removes that optional dependency and defers its import until the unused energy-analysis command is
called. No model architecture, training math or sampling code is changed.

### Development evaluation

The development image includes the same pinned LeMat-GenBench source and configuration as the
final tests, with the prepared scorer assets mounted read-only. The agent may score generated
batches and inspect the reports as often as its budget allows. The evaluator uses:

- LeMat-GenBench commit `58e6eae3...`;
- the task's scheduling-only eight-GPU worker patch, parity-checked against a one-worker run;
- fixed UMA pre-scoring variable-cell relaxation, `fmax=0.02`, at most 200 steps;
- the unchanged scorer's additional 50-step ensemble relaxation diagnostics;
- a small-batch guard: Fréchet distance is unavailable with fewer than two valid structures,
  avoiding a NaN-covariance hang; SUN/MetaSUN and well-posed metrics are unchanged;
- ORB `orb_v3_conservative_inf_omat` with the `orb_conserv_inf` hull;
- MACE `mp` with the `mace_mp` hull;
- UMA `uma-s-1`, task `omat`, with the `uma` hull;
- LeMat-Bulk `compatible_pbe` at revision `0dc17eea...` and the MLIP hull snapshot at `70d505bb...`;
- structure-matcher novelty and uniqueness, made disorder-aware by `lemat-substitution-novelty.patch`
  (`lemat_genbench/fingerprinting/substitution.py`): a structure is not novel when merging a dilute
  substituent into its host gives a known LeMat-Bulk structure, and two dilute variants of the same
  parent are one structure. Substitutable pairs follow MatterGen's modified Hume-Rothery rule
  (relative atomic-radius difference ≤ 0.3, electronegativity difference ≤ 1.0); dilute means the
  substituent holds at most a quarter of the pair's sites. Why: an agent session reached SUN +
  MetaSUN 0.78 (baseline 0.21) by placing one similar element on a site of known compounds (for
  example GdDy₇Cu₈ from DyCu); those ordered solid-solution approximants were 81 % of its
  structures. Of MP-20 test materials, 8 % are such a variant of another LeMat-Bulk entry (23 %
  at a one-third share, 43 % at one half); of the OMatG starter's structures, 1.2 %;
- stable threshold 0.0 and metastable threshold 0.1 eV/atom.

One upstream subtlety matters: the pinned SUN implementation thresholds the ensemble
`e_above_hull_mean` of the geometry **entering LeMat**, not the geometry from its internal
relaxation diagnostics. CrystalGen first relaxes the sealed candidate with UMA-200, then passes
that geometry into the unchanged LeMat scorer. The headline therefore measures the relaxed
structures without changing the leaderboard's metric implementation.

`bash /workspace/evaluation/score.sh INPUT RAW_OUTPUT SCORE_DIR` runs the complete pipeline;
`relax.sh` runs the relaxation alone. See the
[development guide](environment/evaluation/README.md). The visible example requests 100
structures; the final batch has 2,500, so small-batch scores are noisier and batch-level uniqueness
can differ. Agents may run 2,500-sample development batches too.

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
never imports or executes submission code. It preserves the original snapshot, relaxes a separate
copy, then runs one global LeMat batch. Neither a submission report nor a flag can skip or change the
relaxation. Failed refinements retain the original bytes; finite unconverged iterates are kept and
reported. Missing outputs remain missing.

| Final track | Inputs | Evaluation |
| --- | --- | --- |
| Unconditional | One verifier-chosen request totaling 2,500 structures with a fresh seed | UMA-200 pre-relaxation, then exact pinned comprehensive multi-MLIP hull scoring |

`reward.json` holds `reward` = `unconditional_sun_metasun` = (SUN count + MetaSUN count) / 2,500,
already in [0, 1]. This matches the public leaderboard's primary ordering after normalizing its
count by the submitted set size. The other official funnel metrics are separate keys; there is no
task-invented weighted aggregate. `report.json` (`crystalgen-report/v2`) records the relaxation
recipe, convergence and fallback counts and stage runtimes; relaxed CIFs and per-sample logs stay in
the grader's results directory.

### Metrics

- `unconditional_sun_metasun` (= `reward`): fraction of all 2,500 requested structures that are
  either SUN or MetaSUN. The headline leaderboard-comparison metric.
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
2,000-epoch starter took 5.4 h on six H100s. Scoring 2,500 structures took about 1.2 h of
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

## 6. Results and validation

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

- About half of the starter's reward is reached by the no-learning smoke generator: the fixed UMA
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
