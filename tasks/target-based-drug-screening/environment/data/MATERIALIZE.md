# Prepare a run

These commands run on the controller, before the offline agent starts.
Run from target-based-drug-screening/. For the shared runner, use the preparation command in the
[task README](../../README.md#5-running). For manual preparation, install
`environment/data/requirements.txt` in a controller environment.

1. Materialize and validate the existing compact data release:

       export TARGET_BASED_DRUG_SCREENING_DATA_DIR=/large-disk/drugclip/materialized
       bash environment/data/prepare.sh full

   The default release is pinned to the GPU-trained reference data:
   c8d2febc996e4c96b10f3aecd89fb7375150449f. Verify that
   HF_MATERIALIZATION.json records this SHA and the expected repo_id.
   TARGET_BASED_DRUG_SCREENING_HF_REVISION can explicitly override it,
   including main for discovery; a changed release is a new comparison. Source revisions/hashes and the rebuilding
   protocol remain documented in [PIPELINE.md](PIPELINE.md). See
   ../../docs/REPRODUCIBILITY.md for the controller's complete run record.

2. Prepare the generic Uni-Mol initialization:

       python environment/data/prepare_pretrained.py /large-disk/drugclip/pretrained

   This downloads three pinned assets, validates their sizes and SHA256
   hashes, and writes manifest.json. It never fetches benchmark-trained
   DrugCLIP weights. Existing correct files are reused.

3. Revalidate the source and create fresh disjoint views:

       PYTHONPATH=environment/data python -m drugclip_data.cli validate --data-root /large-disk/drugclip/materialized
       python environment/data/stage_task_data.py \
         --data-root /large-disk/drugclip/materialized \
         --pretrained-root /large-disk/drugclip/pretrained \
         --development-root /large-disk/drugclip/development \
         --private-root /large-disk/drugclip/private

   Destinations must not exist. Development receives only the three
   training/dev LMDBs, their manifest, split IDs, and the three allowed
   pretrained files plus manifest. Private receives only the 117 targets'
   labeled molecule and pocket LMDBs. Raw archives, test labels/metadata, and
   author benchmark-trained weights cannot enter the development view.
   Each view receives .complete.json with its source identity, view name,
   relative file paths, byte sizes, SHA256 hashes, and file-manifest digest.
   Keep the private receipt with the controller. Use fresh staged views
   when the source changes and retain the old receipts for comparison.
   Views use real copies, so reserve extra disk roughly equal to the
   processed training files + test LMDBs + 191 MB of initialization assets.
   The existing full-download launcher conservatively requires 90 GiB free
   for materialization/extraction; account separately for view copies.

4. Set the Harbor mounts and a fresh research directory:

       export TASK_DATA_DIR=/large-disk/drugclip/development
       export TASK_PRIVATE_DIR=/large-disk/drugclip/private
       export TASK_RESEARCH_DIR=/large-disk/drugclip/research-run-001
       mkdir -p "$TASK_RESEARCH_DIR"

   The agent mounts only processed/, splits/, pretrained/ and their development
   receipt .complete.json from TASK_DATA_DIR.
   The verifier mounts private/test beneath a root-owned mode-0700 /private
   directory. Its inference child runs as UID 65534 and receives fresh,
   field-whitelisted LMDBs with label fields removed, molecules privately
   permuted per target and fresh numeric indices. The original-index map
   and seed remain root-only in the verifier's run record. Pocket order
   remains unchanged. Start sessions through the shared Harbor runner; see
   ../../docs/RUN_HANDOFF.md and the task README for its entry points.

The controller image moved to environment/Materialize.Dockerfile and its
compose to environment/data-compose.yaml. prepare.sh and run-data.sh retain
their previous entry points. environment/Dockerfile and docker-compose.yaml
now define the agent workspace. Neither image includes large datasets.
