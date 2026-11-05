# Independent verifier

The controller snapshots the current submission and runs tests/test.sh in a
separate offline Harbor environment. tests/docker-compose.yaml uses the task
root as its build context and mounts TASK_PRIVATE_DIR/test under /private.
The image creates /private and /logs/verifier as root-only directories.

The grader:

1. requires root privileges and an inaccessible private-data ancestor;
2. checks exactly 102 DUD-E and 15 LIT-PCBA targets;
3. validates labels and creates fresh field-whitelisted, label-free LMDBs;
   molecules are independently shuffled per target with fresh numeric keys;
4. copies the candidate snapshot and makes the package/inputs read-only;
5. runs infer.sh with supplementary groups cleared, UID/GID 65534, a clean
   environment, no network, and a 22-hour inference timeout;
6. privately maps shuffled indices back to trusted labels, scores with the same metric module used
   by development evaluation and writes /logs/verifier/reward.json.

No candidate code runs with grader privileges. The trusted scorer and output
directory are unavailable to it. Candidate errors produce reward 0;
infrastructure errors produce no reward. The v2 scalar is the mean of the
two dataset-mean BEDROC(alpha=80.5) values, with equal dataset weights and
equal target weights within each dataset. Both complete datasets are required.
All raw metrics are retained; no nonlinear mapping is applied.

For a controller-side replay with the image built locally:

    docker build -f tests/Dockerfile -t drugclip-verifier:local .
    docker run --rm --network none --gpus 1 \
      -v "$TASK_PRIVATE_DIR/test:/private/benchmark/test:ro" \
      -v /path/to/submission-snapshot:/workspace/submission:ro \
      -v /path/to/verifier-output:/logs/verifier \
      drugclip-verifier:local bash /tests/test.sh

The host must expose the contract's one inference GPU. The command uses the
host GPU runtime; Harbor normally applies task.toml's GPU/network settings.

Unit and synthetic integration checks are in test_agent_task.py. They include
real LMDB redaction, complete/partial/duplicate coverage, tie invariance,
atomic publication, privilege-separated replay, and corrupt-data failures.
They do not establish scientific performance or GPU/container acceptance.
The permutation seed and view-to-original .npy maps live under root-only
/logs/verifier/private-order. They are retained for controller reproducibility
and never copied into inference inputs. `reward.json` contains the reward
`dataset_balanced_bedroc`, and six equal-target mean screening metrics for a valid package. An invalid
package receives only a zero reward. The shared runner collects `report.json`
with scoring_protocol=drugclip-dataset-bedroc/v2, per-target scores, benchmark
means or the candidate error. Existing readers
can use the identical `details.json`. The report contains the private manifest's
digest. Only the controller retains its seed and permutation maps.
Index-only predictions lose the actives-first advantage in the synthetic
regression; the real released order still needs an audit.
For controlled comparisons, grade.py accepts --permutation-seed-file pointing
to a private 32-byte seed behind a root-only boundary. Otherwise fresh private
entropy is generated each run. Keep mapping/seed files private when sharing
metrics. Standalone score.py supports --private-mapping; source-order reference
control scoring remains available when that option is omitted.
Record the actual verifier runtime using /tests/record_environment.py before
candidate code runs; see ../docs/REPRODUCIBILITY.md. Session orchestration and
snapshots use the shared runner described in ../docs/RUN_HANDOFF.md.

For trusted archived full-suite reports, `python tests/reaggregate.py old.json new.json`
recomputes the v2 reward without inference. It refuses to overwrite any output,
requires every target, ignores cached aggregate values, and records the source
hash and original reward. Do not use this helper on candidate-supplied reports.
