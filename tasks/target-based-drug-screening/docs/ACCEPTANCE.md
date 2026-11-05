# Drug-screening acceptance, 2026-11-04

The original runs below used the legacy EF-derived v1 reward. Current task
version 0.3.0 uses dataset-balanced BEDROC (drugclip-dataset-bedroc/v2);
see the v2 migration section at the end for current scores and checks.

## Workspace and release

The agent and verifier Dockerfiles both built from the task root. Harbor 0.23
loaded schema 1.3. The agent receives read-only development LMDBs, split IDs,
generic Uni-Mol assets and the development receipt. Private LMDBs are mounted
only under the verifier's root-only `/private`. The benchmark-trained epoch-14
weights are for controller-side reproduction only.

The agent prompt has the five sections and the developer README has the eight
sections in the repository standard. The maintained model lives in
`environment/baseline/drugclip/`. The atom-histogram starter is under
`environment/references/minimal-starter/`. Historical source paths resolve to
these copies. The normal preparation and verification commands use the
[shared runner](../../../runner/README.md).

| Check | Evidence |
| --- | --- |
| Development versus private data | The staging script hashed every allowlisted file. The private view contained 102 DUD-E and 15 LIT-PCBA targets. The agent passed the Compose health check with no test tree. |
| Shared preparation | Starting from an empty controller root, `harness/agent-eval/prepare.sh` fetched the pinned release, validated and staged both views and wrote the runner environment file. Both file manifests matched those used for the verification runs byte for byte. |
| Agent image, offline | Both short commands in the starter READMEs trained on real development records, exported a standalone package and published into the same live submission directory. The second package's weights matched the selected export by SHA256. |
| Training check | CPU, seed 1, one epoch on eight records at each training stage and eight dev records. The 95,305,418-parameter model completed four updates per stage, reaching dev BEDROC 0.5000 on that tiny slice. |
| Regression suite | 61 tests passed in the pinned task image with Python 3.11.13 in 24 s, with no skipped tests, including actual UID/GID 65534 inference, rejected partial submissions, equal-score ties, synthetic permutation checks and publication on OverlayFS. |

A Harbor copy-agent trial replayed the atom-histogram package on one RTX A6000.
The full trial took 6 min 26 s, including separate agent and verifier startup.
It scored **EF1% 1.602231** and **legacy v1 reward 0.615714** over all 117 targets and
4,241,148 ligand records. Harbor read all seven numeric fields. This score
belongs to the eight-record starter, independently of the author's epoch-14
model. The trial capped the copy environment at one GPU.

The original independent verifier's v1 scalar was `reward = E/(1+E)`, where `E` is the
equal-target mean EF1%. Valid `reward.json` files also contain the six
`overall_mean_*` metrics. The shared runner retains per-target scores and
benchmark means in `report.json`. Existing consumers can still read
`details.json`.

## Released-baseline results

Two independent full-suite replays recovered the author's EF1% result. Both
covered all 117 targets and 4,241,148 ligand records using the same fixed
checkpoint. The second run passed through Harbor 0.23 and the task's pinned
agent and verifier Dockerfiles. It took 90 min 2 s on one shared RTX A6000.

| Benchmark | Targets | Author EF1% | Independent EF1% |
| --- | ---: | ---: | ---: |
| DUD-E | 102 | 18.055988235982 | 18.055988235982 |
| LIT-PCBA | 15 | 2.861166091213 | 2.861166091213 |
| Equal-target combined | 117 | 16.107934114858 | 16.107934114858 |

The verified legacy v1 reward is **0.9415475887803699**. The two private-permutation
manifests have different hashes. All four EF cutoffs agree per target to
floating-point precision. The largest per-target difference was 2.76e-5
for ROC-AUC and 4.14e-6 for BEDROC across the two inference runtimes.
The aggregate record includes the exact source revisions, checkpoint hash,
coverage, six metrics, benchmark means, raw-report hashes and these differences.
It contains aggregate scores and digests. The controller retains the private
permutation maps. [Reproduction record](../environment/baseline/drugclip/reproduction/independent_replay.json)

## Replaying the author's selected weights

The author released epoch 14 with its model configuration and manifest.
Its SHA256 matches both the published manifest and the downloaded 381,378,659
bytes. The checkpoint loaded strictly into the repository's 95,305,418-parameter
dual Uni-Mol model. The original training seed, raw training logs and exact
training-code revision have not been supplied. [Released weights](https://huggingface.co/chromeNLP/openssi-target-based-drug-screening/tree/1f4e4a2ca0784438c4d69ccea5ffb27fb065f68a)

To assemble the controller's replay package, run the following in the task
directory. Use a separate fresh directory outside the agent's data and research
mounts. The release omits the inference scripts and the training summary. We
use the reference exporter files and set its required pocket-crop seed to 1,
the default in this task. The historical crop seed is unconfirmed.

```bash
pkg=/large-disk/controller/drugclip-epoch14
export HF_HOME=/large-disk/controller/huggingface-cache
mkdir -p "$pkg"
hf download chromeNLP/openssi-target-based-drug-screening \
  --revision 1f4e4a2ca0784438c4d69ccea5ffb27fb065f68a \
  README.md model.pt model_config.json model_manifest.json --local-dir "$pkg-release"
cp "$pkg-release"/{README.md,model.pt,model_config.json,model_manifest.json} "$pkg/"
(cd "$pkg" && echo 'bb108dcec5608b04eee546a0ed57248ab5f956cb61799a8c5611ab9c8da7c772  model.pt' | sha256sum -c -)
cp environment/baseline/drugclip/{infer.sh,predict.py,model.py,data.py,common.py,training.py,validation.py,unimol_encoder.py,requirements.txt} "$pkg/"
printf '%s\n' '{"pocket_crop":{"policy":"author_distance_weighted_v1","seed":1}}' > "$pkg/training_summary.json"
```

Score this package with `run_verify.sh` from the [task README](../README.md#5-running).
The pinned dataset revision is `c8d2febc996e4c96b10f3aecd89fb7375150449f`.
The author reported mean EF1% 16.107934114857695. The independent replay uses
the inference and scoring code from upstream revision
`1ddf16992756167aacf8ba647aa17c5ffac0faaa`, a fresh private molecule
permutation, and an RTX A6000. Its Python 3.12.11 environment used PyTorch
2.7.1+cu126, RDKit 2025.03.3, NumPy 2.2.6 and LMDB 1.6.2.

The regression command from the task root was the following, with the data
test dependencies and pytest 8.4.2 installed in a test-only directory. The
container ran offline as root so privilege-drop checks executed. Synthetic
test scratch used a 2 GiB tmpfs.

```bash
PYTHONPATH=environment/data python -m pytest -q -rs -p no:cacheprovider \
  tests environment/evaluation/tests environment/baseline/drugclip/tests \
  environment/references/minimal-starter/tests
```

An eight-H100 research session and H100 inference timing are outside the scope
of this machine's acceptance run.

## Workspace audit follow-up, 2026-11-05 UTC

The prompt now identifies the independently verified reference and explicitly
describes target weights and the scalar's meaning. The current scalar is a
controller normalization, not paper accuracy or percent solved. Its definition
was unchanged in that packaging audit; the later v2 migration is recorded below.

The agent image now includes only public ranking mathematics, a runnable BioLiP
development evaluator and its instructions under `evaluation/`. The private
label reader and permutation builder remain controller-only. The grader imports
the extracted `ranking_metrics` function; its Python AST is identical to the
previous implementation. No inference model or metric definition was changed,
so the two recorded full-suite replays remain applicable; they were not repeated
for this packaging change.

- Both final images rebuilt. The actual Compose agent became healthy with
  `network_mode: none`. In-container probes verified that the test tree,
  verifier and ordering code were absent, the five development mounts were
  read-only even to root, no benchmark checkpoint was baked into baseline or
  references, no credential variables were injected, and an external TCP
  connection failed. The standard layout and mount checker also passed.
- All 61 regression tests passed in 21.06 s in the final verifier image, with
  no skips. The real privilege-drop tests additionally checked empty
  supplementary groups, zero effective capabilities, secret-environment
  filtering, denied input/package writes and chmod, unreadable verifier and
  private mapping, and writable prediction/scratch directories.
- The standalone dev command successfully loaded the existing eight-record
  wiring model in the final offline agent image. On the same eight development
  records with batch size four it produced BEDROC 0.5000, AUC 0.58333 and eight
  evaluated queries, matching the earlier training smoke's BEDROC. No weights
  were updated. Full-development scoring is available by removing the limit.

These checks cover task packaging, mounts and the inference privilege boundary.
Dataset preparation also validates the pinned split metadata and similarity
filters. This audit does not establish absence of overlap in the generic
Uni-Mol checkpoint's original pretraining population.

## BEDROC v2 migration and final checks, 2026-11-04 (America/New_York)

Task version 0.3.0 adopts `drugclip-dataset-bedroc/v2`. The primary reward is
0.5 times the mean target BEDROC(alpha=80.5) on each of DUD-E and LIT-PCBA.
Both datasets must be complete. The six original equal-target raw diagnostics
are retained. Valid reward files contain eight numeric fields; detailed
reports carry the metric version. Invalid submissions score zero, while
infrastructure failures still leave no reward.

`tests/reaggregate.py` migrated all three available full-suite reports into
new files, retaining source hashes and old rewards. The original numerical
replay records and development curve remain byte-identical. The baseline
rewards are **0.166384906515** (Harbor) and **0.166385032576** (upstream).
The historical wiring package re-aggregates to **0.028706422861**.
[Versioned aggregate receipts](../environment/baseline/drugclip/reproduction/independent_replay_v2.json)

Final validation:

- **71 tests passed**, no skips, in 21.50 s in the final verifier image.
  New cases verify equal dataset weights despite unequal target counts,
  perfect/worst endpoints, incomplete/duplicate targets, invalid values,
  recomputation from per-target records, and refusal to overwrite originals.
  Actual UID/capability/read-only/private-map checks remain enabled.
- A fresh offline Compose workspace passed the standard prompt/README,
  task schema/resource and mount review. It contained neither test data nor
  private scoring/orchestration code. The final Dockerfile copies an explicit
  list of model files and aggregate reproduction records, excluding operator
  launchers, historical checkpoint evaluators, audits and source tests.
  Submission and research start empty; no model weights, LMDBs, caches or
  compiled Python files are baked into the workspace. Data mounts reject
  writes even from root; credential variables and external connectivity are
  absent in the test workspace.
- The real 95M reference trained on eight records per stage in that workspace,
  completed four updates in each stage, exported, ran standalone development
  evaluation (eight queries, BEDROC 0.5000), and published twice successfully.
  This is a CPU wiring check, not a new scientific baseline training run.
- **Native Harbor 0.23 passed** with the existing atom-histogram package under
  the new verifier. It evaluated all **117 targets and 4,241,148 ligand
  records**, returning **reward 0.028706453638**, with all eight numeric fields
  read by Harbor. Wall time was 6 min 13 s on one shared RTX A6000. The reward
  differs from re-aggregating its earlier replay by 3.08e-8; EF1% agrees to
  floating-point precision. [Run receipt](harbor_bedroc_v2.json)
- The task website uses the same v2 formula for the verified baseline, author
  report and short agent. The baseline displays **16.64/100**; the old receipt
  remains downloadable. All 98 site tests, 105-page navigation checks and
  22-track page/export checks passed, including live HTTP verification.
- The temporary audit container and Harbor agent/verifier/egress containers
  were removed. Run evidence remains on the large disk. No unrelated running
  workloads or their data were removed.

The ranking metric implementation is unchanged; full baseline inference was
not repeated for an aggregation-only migration. Scientific controls concerning
ligand-only predictions, mismatched pockets and pretraining overlap remain
outstanding as described in [the metric protocol](METRIC_DESIGN.md). These
successful format and isolation checks are not a proof of an unhackable task.

## Shared runner cleanup, 2026-11-05 UTC

The task-local `controller/` and its standalone reference driver were removed
in favor of the repository's shared Harbor runner, as required by
`CONTRIBUTING.md`. Eleven tests specific to that retired runner were removed;
the model, scoring, data isolation and publication tests remain. Historical
validation entries retain their original counts and are labeled accordingly.

All 60 remaining tests passed in the pinned verifier image in 19.13 seconds,
with no skipped tests, including real UID-separated inference. The layout
check passed for the five prompt sections, eight developer README sections,
required files, schema 1.3 and read-only development/private mount separation.
`git diff --check` also passed. This cleanup changes no agent-image inputs,
training/inference code or metric; the recorded full-suite replays still apply.
