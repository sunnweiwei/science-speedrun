# Reproduce an acceptance run

Run records belong to the controller. Keep one record directory per run,
outside submission/ and outside agent-visible development data. A record
must identify the actual data bytes, images, runtime, run configuration,
and frozen candidate. A mutable branch name or image tag is insufficient.

## Data lock

The compact release defaults to the immutable revision used by the reported
GPU-trained reference:

    export TARGET_BASED_DRUG_SCREENING_HF_REVISION=c8d2febc996e4c96b10f3aecd89fb7375150449f
    export TARGET_BASED_DRUG_SCREENING_HF_REPO=zifeng-ai/target-based-drug-screening

Materialize and verify that HF_MATERIALIZATION.json records this repo_id and
resolved commit. An explicit revision override, including main for discovery,
is supported; record its resolved SHA and treat a different release as a new
comparison. An upstream rebuild instead uses the
committed source_manifest.json, protocol.json, and actual output fingerprints;
do not invent a Hugging Face revision for that mode.

stage_task_data.py validates the permitted initialization assets, then writes
a .complete.json receipt for each staged view. Each receipt records:

- resolved source repository/commit, when available;
- development or private view;
- every staged relative file path, size, and SHA256;
- SHA256 of the sorted file-manifest representation.

The hashes describe the copied bytes. They support comparison of repeated
runs and detection of later modifications; they do not replace the original
data pipeline's scientific contamination validation. Existing views with old
receipts should be re-staged into new directories to obtain fingerprints.

Copy the receipts to the controller's run record. The private receipt stays
outside the agent container:

    cp "$TASK_DATA_DIR/.complete.json" "$TASK_RUN_RECORD/development.data.json"
    cp "$TASK_PRIVATE_DIR/.complete.json" "$TASK_RUN_RECORD/private.data.json"

Keep the original HF_MATERIALIZATION.json and source validation receipt there
as well. Never mount the combined materialization root into the agent.

## Environment lock

Record the checked-out Git commit, Harbor version, sanitized resolved run
configuration, and exact built-image IDs or repository digests. Keep the
Dockerfile base digest as provenance, but also record the final images, since
they include the task code and installed libraries.

record_environment.py collects the observed Python/package versions,
PyTorch CUDA runtime, visible GPU names/memory, and driver information.
It does not dump environment variables or credentials. The Git commit and
image ID are explicitly controller-supplied identifiers.

After the actual run containers start, before candidate code runs, the
controller supplies TASK_AGENT_CONTAINER and TASK_VERIFIER_CONTAINER:

    export TASK_CODE_COMMIT="$(git rev-parse HEAD)"
    export TASK_AGENT_IMAGE_ID="$(docker inspect --format '{{.Image}}' "$TASK_AGENT_CONTAINER")"
    export TASK_VERIFIER_IMAGE_ID="$(docker inspect --format '{{.Image}}' "$TASK_VERIFIER_CONTAINER")"
    python target-based-drug-screening/environment/record_environment.py \
      --role controller --code-commit "$TASK_CODE_COMMIT" \
      --output "$TASK_RUN_RECORD/controller.environment.json"
    docker exec "$TASK_AGENT_CONTAINER" python /workspace/record_environment.py \
      --role agent --code-commit "$TASK_CODE_COMMIT" --image-id "$TASK_AGENT_IMAGE_ID" \
      > "$TASK_RUN_RECORD/agent.environment.json"
    docker exec "$TASK_VERIFIER_CONTAINER" python /tests/record_environment.py \
      --role verifier --code-commit "$TASK_CODE_COMMIT" --image-id "$TASK_VERIFIER_IMAGE_ID" \
      > "$TASK_RUN_RECORD/verifier.environment.json"

Create TASK_RUN_RECORD first. Verifier reporting occurs when its container
starts, which may be after the agent phase. Controller and container reports
are separate: host-installed versions are not evidence for the CUDA images.
A report from a standalone image probe is likewise not proof of the final
job's GPU allocation.

Keep model/agent name and version, seeds, exposed resources, timeout settings,
start/deadline timestamps, and relevant recipe settings in the resolved run
record. Omit authentication values from saved configuration.

## Candidate and replay lock

At handoff, freeze one complete package and validate it. Record hashes for
every owned inference file, plus the model manifest and selected checkpoint.
Store training/dev logs, training_summary.json, agent trajectory, controller
handoff events, and verifier outputs alongside the record. Keep optimizer
states and other candidates under the run's persistent research directory.

Use the same frozen package for any preliminary subset replay and the full
117-target replay. Preliminary metrics must identify the subset explicitly.
The controller records a success only after independent replay produces a
valid result with complete required coverage and finite metrics.

The shared runner keeps the agent trajectory, published package, snapshots and
verifier reports. Keep the additional data receipts, image IDs and runtime
observations above with that run record. See [RUN_HANDOFF.md](RUN_HANDOFF.md)
for the shared entry points. These records are operator-owned, not part of the
agent's submission.

The verifier retains private-order/manifest.json and per-target .npy maps
under its root-only output. The manifest records the private seed, generator,
NumPy version and mapping checksums. For exact replay, retain these privately
and reuse the seed with the same verifier dependencies; never publish them
to an agent alongside labels or benchmark results. The original staged-data
hashes remain unchanged. Index-only random rankings still have sampling
variance; one shuffle does not force every such ranking to EF1%=1.

Reference inference now seeds molecule conformer selection/cropping by the
configured seed and SMILES rather than temporary path/original record index.
The same molecule therefore receives the same preprocessing when privately
reordered. Pocket cropping and the training recipe are unchanged. Historical
metrics used their historical code and are not retroactively rerun results.

The reported A100 training/full-suite score and complete dev curve are stored
under environment/baseline/drugclip/reproduction/. Their Python 3.11 / PyTorch
2.6.0+cu124 environment differs from the declared CUDA images. The supplied
report does not include raw artifacts or prove the current Harbor verifier
path; CPU fixtures, reported scientific results, and container acceptance
remain separate evidence. Future acceptance records should include the
complete run artifacts described above.
