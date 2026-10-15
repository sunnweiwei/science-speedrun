# science-speedrun

Long-running model-development tasks for AI research agents, in the native
[Harbor](https://www.harborframework.com/docs/tasks) task format, with a runner that starts an agent
session on one of them and scores what the agent publishes. Each task gives the agent a dataset, a
reproduced baseline with its logs, development evaluation and a submission contract. The agent
keeps improving a model package at `/workspace/submission` with no end point; an independent
verifier replays that package on held-out data.

The tasks here passed a full agent session: a Codex agent trained, evaluated and published a valid
package, and the verifier scored it. The agent behaved as the instruction intends (no exploits,
no stalled pipeline).

## Verified tasks

The agent column is a behavior test: one GPU and 4–6 h per session, not the task budget.

| task | what the agent develops | budget (`task.toml`) | agent test | baseline | reference | data |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [genome-sequence-to-function](tasks/genome-sequence-to-function) | DNA sequence → 5,930 genomic tracks and variant effects (AlphaGenome-style) | 8 GPU, 48 h | 0.408 | 0.311 | — | HF release, ~390 GB |
| [single-cell-perturbation-prediction](tasks/single-cell-perturbation-prediction) | zero-shot perturbation response in unseen cell lines | 8 GPU, 48 h | 0.151 | 0.103 | — | HF release, ~35 GB |
| [mass-spectrometry-molecule-retrieval](tasks/mass-spectrometry-molecule-retrieval) | MS/MS spectrum → molecule retrieval (Hit@1) | 8 GPU, 48 h | 0.465 | 0.405 | 0.494 | HF release + private labels |
| [pde-surrogate](tasks/pde-surrogate) | autoregressive surrogate for 2-D time-dependent PDEs | 1 GPU, 48 h | 0.503 | 0.345 | — | HF release, ~200 GB |
| [timeseries-forecasting](tasks/timeseries-forecasting) | zero-shot forecasting foundation model on GIFT-Eval | 8 GPU, 48 h | 0.334 | 0.275 | 0.447 | GiftEval from HF, 275 GB |
| [surface-code-decoding](tasks/surface-code-decoding) | neural decoder for Sycamore surface-code experiments | 8 GPU, 48 h | 0.7626 | 0.7557 | 0.7615 | HF release, ~8 GB |
| [natural-disease-progression](tasks/natural-disease-progression) | next-year disease risk from MIMIC-IV histories | 1 GPU, 24 h | 0.243 | 0.236 | 0.268 | credentialed HF release (MIMIC-IV) |
| [probabilistic-global-weather](tasks/probabilistic-global-weather) | 15-day global ensemble weather forecasting model (WeatherNext2-style) from random initialization | 4 GPU, 24.5 h | 0.108 | 0.674 | — | HF release, ~206 GB |
| [protein-folding-prediction](tasks/protein-folding-prediction) | one from-scratch biomolecular folding model for all nine FoldBench tracks (monomers, interfaces, ligands) | 8 GPU, 48 h | 0.0003 | 0.037 | 0.557 | HF release, 73 GB |
| [tokamak-magnetic-control](tasks/tokamak-magnetic-control) | one shared magnetic shape/position/current controller for a MAST-U-like tokamak (FreeGSNKE), sealed randomized plant | 1 GPU + 64 CPUs, 48 h | 0.529 | 0.486 | — | none (machine description vendored) |
| [crystal-structure-generation](tasks/crystal-structure-generation) | unconditional crystal-structure generator trained from scratch on MP-20, maximizing stable/metastable, unique and novel (SUN + MetaSUN) structures under LeMat-GenBench | 8 GPU, 24 h | 0.157 | 0.208 | — | MP-20 + scorer assets (~25 GB; gated `facebook/UMA` needs HF_TOKEN) |
| [molecular-pretraining-transfer](tasks/molecular-pretraining-transfer) | ≤90M molecular encoder pretrained from scratch on the leakage-filtered Uni-Mol2 corpus, transferred to 24 tasks; test MA-RAE QM9 / COMPAS-1D / ExpansionRx / ASAP (lower is better) | 8 GPU, 12 h | 0.186 / 0.164 / 0.735 / 0.657 | 0.183 / 0.166 / 0.744 / 0.636 | 0.192 / 0.121 / 0.668 / 0.644 | rebuilt from pinned public sources (179 GiB download, 173 GiB release) |
| [protein-design](tasks/protein-design) | one all-atom backbone generator for four design tracks; per-track pass rate (designable by LigandMPNN + RoseTTAFold3 refolding and novel vs the fit corpus) enzyme / sm_binder / ppi_binder / unconditional | 1 GPU, 24 h | 0.082 / 0.025 / 0 / 0.10 | — | — | development views (RCSB snapshot, HF, ~6 GB); 51-target evaluation view vendored |
| [protein-sequence-modeling](tasks/protein-sequence-modeling) | one from-scratch protein sequence model with heads for eleven PEER understanding tasks (mean task reward) | 1 GPU, 48 h | 0.647 | 0.670 | — | PEER tasks (TAPE/PEER S3, 1.6 GB) + UniRef50 2026_03 decontaminated with MMseqs2 (9.9 GB) |

Notes on the table:
- Rewards are raw task metrics; larger is better, except molecular-pretraining-transfer's MA-RAE (lower is better).
- Baselines were reproduced in this setup. Their numbers, logs and validation outputs are in each task's `environment/baseline/<name>/` (`RESULTS.md`, `reproduction/`). Agents are told not to retrain them.
- Every verifier finishes within about 2 h on the task's verifier resources, except mass-spectrometry: about 3.5 h on one GPU.
- probabilistic-global-weather was tested on the CMU Babel cluster: a 6 h Codex session on two L40S. The agent checked its
  pipeline and trained a fresh from-scratch candidate on a longer schedule; its package scored 0.108 against the
  starter's 0.674 (13.1 h on four H100s), as expected for the budget. The verifier ran all eight cases and passed in
  7.6 h on three L40S; it now runs the eight cases at once on eight GPUs (about 2.5 h on H100s).
- protein-folding-prediction was tested on the CMU Babel cluster: a 6 h Codex session on one L40S. The agent trained a
  49.2M from-scratch model (parameter audit shipped) and published four packages; at this budget the scorer's
  stereochemistry checks zero the structures (0.0003), as for the starter's monomer tracks. The verifier ran in
  25 min on 4 L40S. The baseline 0.037 predates the CAPRI-geometry interface rule; the reference is ESMFold2-Fast
  under that rule.
- protein-sequence-modeling was tested on the CMU Babel cluster: a 5 h Codex session on one L40. Against matched
  controls the agent adopted a normalized-loss pretraining recipe, a local secondary-structure head and continued
  pretraining, and fixed a resume-shuffle bug; its package scored 0.647 against the starter's 0.670 (11 h on one H100).
  The verifier needs a real privilege drop (it rejects Apptainer's fakeroot), so the package was scored under Docker on
  another host: 17 s. MMseqs2 is not pinned; the Babel corpus kept 37,338,920 sequences vs the documented 37,294,039.
- protein-design was tested on the CMU Babel cluster in two 5 h Codex sessions on one L40S. In the first, near-copies of
  fit-corpus chains passed refolding, so the task gained a Foldseek novelty gate (a design counts only if its max
  TM-score to every fit chain is below 0.8) and a 20-design evaluation view. The second session designed new helical
  topologies and scored 0.082 / 0.025 / 0 / 0.10 (enzyme designs average max TM 0.67 to the fit corpus); the verifier ran
  in 7.0 h on four A6000 (about 1 h on eight H100s). The starter's published numbers predate the gate and the
  20-design view, so no baseline is shown.
- molecular-pretraining-transfer was tested on the CMU Babel cluster: a 5 h Codex session on two L40S. The agent used
  the baseline's logged numbers instead of retraining it, tested a hard-mask objective, a narrower 80.2M pair track and
  a readout encoder against matched controls, sped up its trainer, and published an encoder on par with the 5.1 h,
  8-H100 baseline (better on COMPAS-1D and ExpansionRx). The verifier ran all 24 fine-tunes in 44 min on four L40.
- crystal-structure-generation was tested on the CMU Babel cluster: a 6 h Codex session on two L40S. The agent
  resampled MP-20 toward low-hull, multinary rows and published a from-scratch candidate; it scored 0.157 against
  the 2,000-epoch starter's 0.208 (5.4 h on six H100s). The verifier relaxes all 2,500 structures with UMA and scores
  them with LeMat-GenBench; with several relaxation workers per GPU it ran in 2 h 9 min on four RTX 6000 Ada.
- tokamak-magnetic-control was tested on the CMU Babel cluster (Apptainer instead of Docker): an 11 h Codex session on
  one L40S with 64 CPUs. Its verifier is CPU-bound: 53 min for the `shape_vc` baseline and 90 min for the agent's
  package on 40 CPUs (35 parallel episodes). Repeated scoring of one package varies by about 0.015 (P4 void
  episodes), so a one-episode difference from the baseline is within noise.

## Quick start

```bash
docker build -f runner/Dockerfile.runner -t harbor-runner:0.23.0-bx runner
export EVAL_ROOT=/persistent/speedrun CODEX_PKG=/opt/codex-0.160.1 MOUNT_ROOTS="/data /opt/codex-0.160.1"
runner/selftest.sh                                              # install check, ~2 min, no GPU
runner/prepare.sh surface-code-decoding /data/surface-code      # data + $EVAL_ROOT/envs/<task>.env
AGENT_KIND=codex HOURS=24 runner/run.sh surface-code-decoding   # session, then independent score
AGENT_KIND=claude-code HOURS=48 runner/run.sh pde-surrogate
runner/score_snapshots.sh NAME 6                                # reward over time from the run's snapshots
```

Setup details are in [runner/README.md](runner/README.md): Codex and Claude Code authentication, resources, two-phase scoring and pitfalls.

## Layout

```text
tasks/<task>/        Harbor tasks (instruction, environment, tests, docs); see TASK_STANDARD.md
runner/              one-command sessions, snapshots and scoring (Harbor 0.23, Codex or Claude Code)
TASK_STANDARD.md     the task format: workspace, instruction, verifier, data
```

## Data and licenses

- **Data.** Each task's `environment/data/` records its sources, licenses and pinned revisions. The tasks redistribute no data. `runner/prepare.sh` downloads the prepared releases or runs the task's materializer.
- **MIMIC-IV.** natural-disease-progression needs PhysioNet-credentialed access to MIMIC-IV, and its derived data must not be redistributed.
- **Vendored code** keeps its upstream license next to it.
