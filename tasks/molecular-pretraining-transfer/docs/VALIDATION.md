# Validation (2026-09-29, one host with 8 × H100 80 GB and 208 cores)

## Acceptance

1. **Images.** Built from the task's own files with `docker compose build`:
   - development: `environment/docker-compose.yaml`, context the task directory;
   - verifier `main` and `grader`: `tests/docker-compose.yaml`.

   All three have Uni-Mol 90f52c4 and Uni-Core ace6fae, with the NCCL timeout patch applied. `/tests` and the
   `agent` user are only in the verifier images. `/workspace` (instruction, contract, baseline, references,
   evaluation) is only in the development image.
2. **Data.** `environment/data/materialize.sh` ran in the runtime image and rebuilt the whole release from pinned
   sources, with downloads already present and verified by sha256. Time:
   - source check: 3 min;
   - leakage filter: 16 min;
   - QM9/COMPAS splits: 1.5 min;
   - suite with ExpansionRx/ASAP conformers: 16 min;
   - views and checksums: under a minute, with the files in page cache.

   The filter's per-shard counts and evaluation-key digest equal those of the earlier development build: 995 molecules
   removed, 125,728,125 kept for training. The 112 content digests are pinned in `environment/data/CHECKSUMS.json`.
3. **Unit tests.** `tests/run_unit_tests.sh` passes 19 tests, run in the grader image against the task checkout.
   They cover:
   - layout, instruction sections, `task.toml` and compose mounts;
   - that the shared runtime block and the two `protocol/` copies are identical;
   - that the checksum table covers every view;
   - metrics against hand-computed values and the official RAE formula;
   - the grader on a synthetic snapshot, a tampered snapshot and a missing molecule;
   - package validation (missing checkpoint, bad `encoder.json`, extra entry, symlink) and prediction-file checks.
4. **Reference recipe.** The baseline (official Uni-Mol2 84M, paper recipe, 30k updates) was pretrained earlier in
   5.1 h on 8 H100 with the same code and flags as `baseline/unimol2-ref/pretrain.sh`. It was not re-run for
   acceptance. That run resumed once at update 10,000 after the unequal-shard fix (see RESULTS.md).

   Inside the development container started from `environment/docker-compose.yaml` (release mounted read-only, no
   test inputs present), `pretrain.sh` ran a 20-update smoke on 8 GPUs. It used the epoch-to-shard links and
   `/temp`, and wrote checkpoints. `publish.py` then packaged the result and ran the contract check (83,425,520
   encoder parameters, all weights present). It swapped the package into the image-layer `/workspace/submission`.
5. **Isolated verifier replay** (`tests/docker-compose.yaml`, `main` + `grader`, both `network_mode: none`, release
   from the materializer):
   - The baseline package from `publish.py` gave `valid` 1.0 and 129 reward keys in 30.7 min. Group MA-RAE:
     QM9 0.1828, COMPAS-1D 0.1655, ExpansionRx 0.7439, ASAP 0.6355.
   - Negative control: the released 84M weights declared as `unimol2_164M` were rejected by the encoder check in
     under a minute (164,081,456 parameters, missing layers). The result was `valid` 0 with the reason, and the grader
     exited without scoring.
6. **Numbers.** The replay's per-task MAE equals the development run of the same protocol (`ft-uni2/fast-v2`) on all
   24 tasks to the printed precision. That run had different GPU placement and was built from the pre-release data
   directory, so the fine-tune and the data rebuild are both deterministic.

   The released 84M and no-pretraining rows in RESULTS.md come from that development run of the same code. They
   were not re-run through the verifier; with deterministic fine-tunes, the known-answer end is not needed again.

## Changes after the replay

`protocol/run_suite.py` starts its fine-tune slots across GPUs slot-major instead of GPU-major. This only spreads
the longest tasks (QM9, COMPAS-1D) over different GPUs; each fine-tune is unchanged. In the replay, the three COMPAS
tasks shared one GPU and set the 31-minute tail.

## Earlier development record

These runs used the same official code in a hand-built image, the pre-release data directory and development
scripts (`mpt-v2/` on the development host):

- calibration of the released 84M with the official 40-epoch recipe on the full QM9 splits: HOMO/LUMO/gap test
  agg-MAE 0.0055 Ha;
- the 5-hour pretraining above;
- a 10-task fast evaluation (`fast-v1`);
- the 24-task evaluation that defines this task (`fast-v2`).
