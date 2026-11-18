# Independent label-isolated verification

`tests/docker-compose.yaml` and `tests/test.sh` implement Harbor's **two-container
label-isolated verifier** (`separate` mode):

1. **`main` service (`--phase infer`, untrusted inference container)**:
   - Mounts `/benchmark/suite:ro`, `/benchmark/features.esmc6b_final_int8.jsonl:ro`,
     and `/benchmark/features:ro` (covering the frozen ESMC-6B INT8 `.safetensors`
     payloads such as `features/esmc6b-final-int8/*.safetensors` and
     `features/esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1/*.safetensors`), plus
     `sealed:/sealed` (`:rw`), `grader_results:/results:ro`, and
     `verifier_logs:/logs/verifier`, with `network_mode: none`.
   - **Never mounts `/benchmark/ground_truths`**. `grade.py --phase infer` also
     refuses to start if `/benchmark/ground_truths` exists.
   - Audits `/workspace/submission` (rejecting escaping symlinks, historical
     `candidate-vN` / optimizer artifacts, and byte-identical copies of the
     supplied ESMFold2-Fast and ESMC-6B checkpoints), executes optional `setup.sh`
     and required `infer.sh` inside `main` only, verifies that predicted outputs
     contain **zero symlinks (`os.path.islink`) and zero path traversal
     (`resolve().is_relative_to`)**, copies regular `.cif` files + SHA-256
     manifest into `/sealed/run`, and signals `READY`.
2. **`grader` service (`--phase score`, trusted scoring container)**:
   - Mounts `/benchmark/suite:ro`, `/benchmark/ground_truths:ro`,
     `sealed:/sealed:ro` (**read-only**), and `grader_results:/results`, with
     `network_mode: none`.
   - **Never mounts, imports, or executes `/workspace/submission`, `setup.sh`, or
     `infer.sh`**.
   - Re-scans `/sealed/run` for symlinks, non-regular files, path escapes, and
     SHA-256 mismatches *before* reading any `.cif` or invoking the pinned
     OpenStructure / DockQ v2 FoldBench evaluator, then writes `report.json`
     (`"label_isolated_sandbox": true`) and `reward.json` atomically to `/results`
     for `main` to publish into `/logs/verifier`.

(A legacy single-process `--phase all` mode with `"label_isolated_sandbox": false`
is retained strictly for offline developer unit tests.)

## Scores

`reward.json` outputs `"score"` and `"reward"` (equal to the V3 5-modality equal
mean across `FOLD_GROUPS`), alongside each individual track's score on `[0, 1]`:

| Modality (`FOLD_GROUPS`) | Constituent Tracks | Track Metric |
| --- | --- | --- |
| `Protein monomers` | `monomer_protein` | Mean all-atom lDDT (`[0, 1]`), missing/nonfinite zero |
| `Nucleic-acid monomers` | `monomer_rna`, `monomer_dna` | Mean all-atom lDDT (`[0, 1]`), missing/nonfinite zero |
| `Protein interfaces` | `interface_protein_protein`, `interface_antibody_antigen`, `interface_protein_peptide` | Fraction with `DockQ >= 0.23` **and** CAPRI acceptable geometry (`LRMSD <= 10 Å` or `iRMSD <= 4 Å`) |
| `Protein–nucleic acid` | `interface_protein_rna`, `interface_protein_dna` | Fraction with `DockQ-v2 >= 0.23` **and** CAPRI acceptable geometry (`LRMSD <= 10 Å` or `iRMSD <= 4 Å`) |
| `Protein–ligand` | `interface_protein_ligand` | Fraction with `RMSD < 2.0 Å` **and** `lDDT-PLI > 0.8` |

## Local interface checks

From the repository root, in a CPU Python environment (`CUDA_VISIBLE_DEVICES=""`)
with PyTorch, Safetensors, PyYAML, and Biopython:

```bash
CUDA_VISIBLE_DEVICES="" TMPDIR=/tmp python -m unittest discover \
  -s protein-folding-prediction/tests -p 'test_*.py' -v
```

These tests exercise the two-phase label-isolated sandbox, symlink/canary escape
defenses, `docker-compose.yaml` mount invariants, CAPRI anti-collapse geometry
gate, and exact `< 1e-12` parity with `harness/ssi_index_v3.folding()`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

- **Upstream Benchmark & Scorers**:
  - *FoldBench: An all-atom benchmark for biomolecular structure prediction* (*Nature Communications*, 2025, `10.1038/s41467-025-67127-3`, pinned commit `4273f6877d82bd0b2fa476d1b2f34d121cbccc70` with OpenStructure 2.8.0).
  - *DockQ: A Quality Measure for Protein-Protein Docking Models* (Basu & Wallner, *PLOS ONE*, 2016) and *DockQ v2* (Mirabello & Wallner, *Bioinformatics*, 2024) for protein and protein–nucleic acid interfaces.
  - *CAPRI: Critical Assessment of PRedicted Interactions* (Lensink et al., *Proteins*, 2007–2023) for macromolecular docking acceptability criteria.
- **Primary Metric (`score` / `reward`)**: Equal-weighted mean across the five canonical biomolecular structural modalities (`FOLD_GROUPS` in `harness/ssi_index_v3.py`):
  $$\text{score} = \frac{1}{5}\Big(\underbrace{S_{\text{prot}}}_{\text{Protein monomers}} + \underbrace{\tfrac{1}{2}(S_{\text{rna}} + S_{\text{dna}})}_{\text{Nucleic-acid monomers}} + \underbrace{\tfrac{1}{3}(S_{\text{pp}} + S_{\text{ab}} + S_{\text{pep}})}_{\text{Protein interfaces}} + \underbrace{\tfrac{1}{2}(S_{\text{prna}} + S_{\text{pdna}})}_{\text{Protein–nucleic acid}} + \underbrace{S_{\text{lig}}}_{\text{Protein–ligand}}\Big).$$
  Grouping the nine FoldBench tracks into five equal-weight structural modalities prevents protein–protein sub-tracks (general PPI, antibody–antigen, protein–peptide) from receiving $3\times$ the weight of all protein monomers or all protein–ligand complexes, matching multi-modal biomolecular folding evaluation practice in FoldBench (2025) and AlphaFold 3 (Abramson et al., *Nature*, 2024).

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **Retention of CAPRI Acceptable Geometry (`LRMSD <= 10 Å OR iRMSD <= 4 Å`) Without `fnat >= 0.1` Relaxation**: Because continuous DockQ averages three terms, $\text{DockQ} = \frac{1}{3}\big(f_{\text{nat}} + \frac{1}{1 + (\text{iRMSD}/1.5)^2} + \frac{1}{1 + (\text{LRMSD}/8.5)^2}\big)$, superimposing or collapsing all chains onto the same volume forces every inter-chain atom pair within the contact cutoff ($f_{\text{nat}} \to 1.0$), yielding $\text{DockQ} \approx 0.33\text{–}0.36 \ge 0.23$ despite physically catastrophic steric overlap ($\text{iRMSD} \approx 15\text{ Å}$, $\text{LRMSD} \approx 30\text{ Å}$). Requiring `DockQ >= 0.23` **and** CAPRI's geometric acceptability bound (`LRMSD <= 10.0 Å` or `iRMSD <= 4.0 Å`, without an `OR fnat >= 0.1` loophole) directly enforces Basu & Wallner (2016) / Lensink et al. acceptability while eliminating chain-collapse gaming, with zero penalty to genuine docking poses (`native` oracle unchanged at `0.985`).
- **Fixed Denominator Including Missing/Failed Targets**: Every target in `suite/targets/<track>.csv` remains in the track denominator (`total / len(targets)`). Omitting OOM, timeout, or stereochemically invalid mmCIF outputs would allow selective abstention on large or difficult assemblies (~7,000 tokens).
- **Top-1 Model-Ranked Selection vs. Oracle Best-of-5**: When multiple seeds/samples are produced, `grade.py` selects strictly the single prediction with the highest model-emitted `ranking_score` per target (`track_score`), never ground-truth `max(DockQ)` or `max(lDDT)` Oracle Best-of-5 selection, because autonomous scientific deployment requires self-consistent confidence ranking without access to experimental coordinates. No extra ad-hoc clash or pLDDT gates are added beyond OpenStructure's native stereochemical/mmCIF parsing and the upstream FoldBench track criteria.

### 3. Diagnostic Signals Retained in report.json

While `reward.json` outputs `"score"`, `"reward"`, and the nine raw track scores, `report.json` preserves rich structural diagnostics:
- `modalities`: exact per-modality scores across the five `FOLD_GROUPS` categories.
- `tracks`: per-track `score`, `targets` (denominator), `finite_targets`, and `missing_or_invalid` counts across all nine FoldBench tracks.
- `expected_predictions`, `accepted_predictions`, `missing_or_invalid_predictions`, and `invalid_predictions` (per-sample mmCIF validation diagnostics).
- `label_isolated_sandbox` (`true` in the two-container `main` -> `grader` pipeline) and `suite_manifest_sha256`.
