# Genome sequence to function (AlphaGenome scale)

## 1. Overview

The agent develops one model that maps 1,048,576 bp of human DNA to 5,930 functional-genomics tracks
in eleven output families:
- RNA-seq, CAGE, PRO-cap, ATAC, DNase, TF and histone ChIP-seq;
- splice sites, splice-site usage and splice junctions;
- contact maps.

The model also predicts the effects of DNA variants and perturbations. This is the AlphaGenome
problem on AlphaGenome's public training data and fold split (`FOLD_0`), using every eligible
training interval.

**Reward.** `overall_mean_family_pearson` on 1,000 held-out test intervals: per-track Pearson in
`asinh` space, averaged within families, then equally across the eleven families.

**Reported next to it** (`report.json` → `variant_effects`, raw metrics, not in the reward), each
with published anchors on the same rows:
- AlphaGenome's own evaluations: eQTL sign, causality and effect size; sQTL; ClinVar non-coding,
  splice-region and missense; MFASS; caQTL; SPI1 bQTL; CRISPRi enhancer–gene linking;
- TraitGym Mendelian and complex-trait causal variants, with Borzoi, Enformer, CADD and GPN-MSA.

**Status.** Integrated and verified on 2026-10-01; see section 6. The integration rebuilt the data,
evaluation and baselines; `docs/VALIDATION.md` records what changed and why.

## 2. Data

| part | content | size |
|---|---|---:|
| train | 41,020 intervals (all eligible in folds 2–7), 8-bit targets | 330.7 GB |
| validation | 200 intervals (fold0), full precision | 4.4 GB |
| test | 1,000 intervals (fold1), full precision, verifier only | 21.8 GB |
| reference | GRCh38.p13 FASTA | 3.3 GB |
| variant evaluations | test: 6,587 perturbations (public queries, verifier-only labels); development split: 2,697 perturbations with labels | < 0.1 GB |

**Sources.**
- AlphaGenome TFRecords: `gs://alphagenome-datasets/v1/train`, public.
- Borzoi fold BED.
- AlphaGenome evaluation tables (CC-BY 4.0).
- TraitGym (MIT).
- GENCODE v46.

Everything is pinned by commit or checksum.
- **Default.** `materialize.sh release` installs the prepared release from Hugging Face
  ([sunweiwei/ai4sci-genome-sequence-to-function-data](https://huggingface.co/datasets/sunweiwei/ai4sci-genome-sequence-to-function-data)).
  It verifies every archive and runs the full validator.
- **Rebuild from upstream.** `materialize.sh rebuild` streams 8.2 TiB and takes about 3 h on
  208 cores.

Details are in [environment/data/MATERIALIZE.md](environment/data/MATERIALIZE.md).

## 3. Workspace

| path | content |
|---|---|
| `/workspace/data/` | reference, manifests, track metadata, train/validation targets, labels and scoring queries, public perturbation queries ([environment/data/README.md](environment/data/README.md)) |
| `/workspace/baseline/alphagenome-reference/` | the verified baseline: reference model code (`reference/alphagenome_50m/`) with multi-GPU training, `predict.py`/`infer.sh`, `perturb.py`/`perturb.sh`, `package.sh`, `publish.sh`, README and RESULTS |
| `/workspace/references/` | AlphaGenome, Borzoi, Enformer and TraitGym, not verified here |
| `/workspace/evaluation/` | `score.py` (tracks), `score_variants.py` (development variant split), `scoring.py`, `variant_scoring.py`: the verifier's code |

The baseline's reproduced numbers are in section 6.

## 4. Evaluation

The verifier (`tests/`) runs in two phases.

1. **Inference.** The `main` service runs the package's `infer.sh` and then `perturb.sh` as
   uid 10001, offline. It sees only the reference FASTA, `public/test.parquet` and
   `public/perturbations.parquet`. It validates every output, copies both outputs into a sealed
   snapshot and hashes it.
2. **Scoring.** A separate `grader` service never executes submission code. It checks the hashes
   and scores tracks (`scoring.py`) and variant effects (`variant_scoring.py`).

Invalid packages score 0, and infrastructure failures publish no reward.

**Output contract v4** ([environment/contract.md](environment/contract.md)). Splice usage and
junctions are sparse: listed sites or pairs only, and unlisted hidden ones count as zero. A full
test run therefore writes about 12 GB of predictions; the v3 dense outputs needed about 600 GB.

## 5. Running

```bash
bash environment/data/materialize.sh release /data/ag      # once per host, see MATERIALIZE.md
export TASK_DATA_DIR=/data/ag TASK_PRIVATE_DIR=/data/ag TASK_RESEARCH_DIR=/scratch/run-001
harbor trial start -p genome-sequence-to-function -a <agent> -m <model>
```

One materialization serves both sides. The agent's compose file mounts only the development
entries of `TASK_DATA_DIR`. The verifier reads the test targets, labels and variant labels from
`TASK_PRIVATE_DIR`.

Resources (`task.toml`):
- **Agent:** 8× H100, 128 CPUs, 768 GiB RAM, 1 TiB scratch, 48 h. The 215M baseline trained in
  4 h 9 min on this hardware.
- **Verifier:** 1× H100, 6 h limit; the package gets 4 h for `infer.sh` + `perturb.sh`. The 215M baseline completes in 31 min (20 min of inference).

## 6. Results and validation

| system | params | training | validation reward | test reward | variant evaluations |
|---|---:|---|---:|---:|---|
| AlphaGenome (published scores) | — | — | — | — | anchors per evaluation |
| tiny reference (author's recipe) | 0.23M | 3 epochs, 3,200 intervals, 1 GPU | 0.114–0.160 (5 runs) | 0.1285 | near chance |
| large reference ([baseline](environment/baseline/alphagenome-reference/)) | 215M | 14 epochs, all 41,020 intervals, 8 H100, 4 h 9 min | 0.302 | **0.3106** | near chance except sQTL (0.73) and enhancer–gene (0.39 vs 0.20 chance); [RESULTS.md](environment/baseline/alphagenome-reference/RESULTS.md) |

Details, timings and the evaluation design are in [docs/VALIDATION.md](docs/VALIDATION.md).

## 7. Limitations

- **One fold, human only.** Training uses `FOLD_0` and the human tracks; AlphaGenome uses four folds
  and mouse tracks as well.
- **128-bp resolution.** Dense outputs are 128-bp bins, while AlphaGenome is base-resolution for
  several assays.
- **Training targets are quantized.** Validation and test targets are full precision.
- **GTEx RNA-seq tracks are masked in the public data.** eQTL readouts use ENCODE RNA-seq tracks:
  the matching tissue ontology when ENCODE has it, otherwise all tracks. AlphaGenome used GTEx
  tracks, so those anchors are a reference, not an identical protocol.
- **Splicing variant scores use splice-site probability change only.** AlphaGenome's scorer also
  uses usage and junctions.
- **Variant evaluations are subsets.** Seeded and label-stratified; rare-positive sets use 100
  positives : 400 negatives. TraitGym and bQTL are drawn genome-wide, so their loci may lie in
  training windows; the others come from test-fold loci.
- **No pretrained AlphaGenome weights** may be used.

## 8. Citation and licenses

- Avsec et al., *Advancing regulatory variant effect prediction with AlphaGenome*, Nature 649
  (2026). The data and evaluation tables are CC-BY 4.0.
- Linder et al., Borzoi, Nature Genetics 57 (2025), for the fold BED.
- Benegas et al., TraitGym (2025, MIT).
- GENCODE v46.
- Task code: see the repository license.
