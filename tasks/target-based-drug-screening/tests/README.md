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
and never copied into inference inputs. `reward.json` contains both `score` and
`reward` (both equal to `dataset_balanced_bedroc`), `dataset_balanced_bedroc`,
and six equal-target mean screening metrics for a valid package. An invalid
package receives `{"score": 0.0, "reward": 0.0}`. The shared runner collects `report.json`
with scoring_protocol=drugclip-dataset-bedroc/v2, per-target scores, benchmark
means, `swapped_pocket_counterfactual` diagnostics, or the candidate error. Existing readers
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

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary V4 task score (`score` / `reward` in `reward.json`, matching `ssi_index_v3.drug_screening()` within `< 1e-12`) is the dataset-balanced Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic at $\alpha = 80.5$:

$$S = \frac{1}{2}\left(\frac{1}{102}\sum_{t \in \text{DUD-E}} \text{BEDROC}_{80.5}(t) + \frac{1}{15}\sum_{t \in \text{LIT-PCBA}} \text{BEDROC}_{80.5}(t)\right) \in [0, 1]$$

- **BEDROC ($\alpha = 80.5$)**: Defined by **Truchon & Bayly (*J. Chem. Inf. Model.* 2007, 47(2), 488–508)** so that 80% of the BEDROC weight comes from the top 2% of ranked compounds ($\alpha = 80.5$), bounded in $[0, 1]$ and less sensitive to active-to-decoy prevalence distortion than unnormalized enrichment factors ($\text{EF}_{1\%}$). Exact score ties are evaluated analytically as the expected metric over all permutations within each tied block, eliminating order-dependent tie artifacts.
- **DUD-E (102 targets)**: **Mysinger et al. (*J. Med. Chem.* 2012, 55(14), 6582–6594)**, providing broad pocket diversity across 102 protein targets.
- **LIT-PCBA (15 targets)**: **Tran-Nguyen, Jacquemard & Rognan (*J. Chem. Inf. Model.* 2020, 60(9), 4263–4273)**, constructed from dose-response PubChem bioassays via asymmetric validation embedding (AVE) unbiasing to test early retrieval under realistic 2D analog similarity and ultra-low active prevalence.
- **Target-conditioned 3D pocket-ligand protocol & decontamination**: Follows **DrugCLIP (Gao et al., *NeurIPS* 2023 / *Science* 2026)** on the frozen 102 DUD-E + 15 LIT-PCBA pocket/conformer test archives, hardened with unidirectional target-pocket sequence coverage decontamination (`mmseqs easy-search -c 0.8 --cov-mode 1` at `> 30%` sequence identity in `protocol.json`, `mmseqs.py`, `profsa.py`, and `biolip.py`, so multi-domain training chains harboring a homologous test pocket domain with `tcov >= 0.80` are excluded even when `qcov < 0.80`) alongside BioLiP2 test-active `ECFP4 Tanimoto > 0.60` and exact InChIKey filtering.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **Equal dataset weighting (`0.5 * DUD-E + 0.5 * LIT-PCBA`) without nonlinear rescaling**: Pooling all 117 targets equally (`102 / 117` DUD-E vs `15 / 117` LIT-PCBA) would give 87.2% of the score weight to DUD-E, whose property-matched decoys are known to allow 2D ligand-only shortcuts (*Chen et al., PLoS ONE 2019*; *Tran-Nguyen et al., JCIM 2020*). Equal dataset weighting gives LIT-PCBA full parity while preserving BEDROC's native $[0, 1]$ physical scale without arbitrary sigmoid or affine rescaling.
2. **No ad-hoc counterfactual subtraction or hard pocket-specificity gate in the primary score**: Subtracting swapped-pocket BEDROC from the primary score or gating on an arbitrary specificity threshold would conflate genuine cross-pocket pharmacophore transfer (e.g., ATP-competitive kinase scaffolds) with shortcut learning and depart from published virtual-screening benchmarks (*Truchon & Bayly 2007*; *LIT-PCBA 2020*; *DrugCLIP*). Instead, swapped-pocket counterfactual replay is recorded purely as a diagnostic in `report.json`.
3. **No SMILES fingerprinting on ProFSADB peptide pseudo-ligands**: ProFSADB pretraining pairs consist of cleaved protein pocket-fragment interactions whose pseudo-ligands are polypeptide segments rather than small-molecule SMILES; protein homology against test pockets is handled via full parent-chain `mmseqs` target-coverage filtering (`cov_mode = 1`), whereas small-molecule `ECFP4 Tanimoto > 0.60` decontamination applies strictly to real ligands in BioLiP2.

### 3. Diagnostic Signals Retained in report.json

While `score` / `reward` is strictly `0.5 * mean_DUDE(BEDROC_80.5) + 0.5 * mean_LIT_PCBA(BEDROC_80.5)`, `report.json` retains full diagnostic granularity:

- **Per-target & per-benchmark screening metrics**: `roc_auc`, `bedroc_80.5`, `ef_0.005`, `ef_0.01`, `ef_0.02`, and `ef_0.05` across all 102 DUD-E and 15 LIT-PCBA targets, plus equal-target `overall_mean_*` summaries.
- **Swapped-Pocket Counterfactual Replay (`swapped_pocket_counterfactual`)**: Replays candidate label-free inference on deterministically deranged target pockets (`N // 2` half-roster shift separating alphabetically adjacent gene-family paralogs) and records `dude_bedroc_80.5`, `lit_pcba_bedroc_80.5`, `dataset_balanced_bedroc`, `pocket_specificity_delta` (`primary - swapped`), `dude_pocket_specificity_delta`, and `lit_pcba_pocket_specificity_delta` to quantify pocket conditioning versus ligand-only prior scoring without altering `score`.
- **Cryptographic audit digests**: `permutation_manifest_sha256` for both primary and counterfactual private per-target ligand permutations.
