# Reconstruction papers and data protocols: facts, reproduction limits, and experiment decisions

Audit date: 2026-10-01. This is the shared entrypoint before designing experiments, modifying starters, or interpreting results. Paper facts and local plans are recorded separately; splits/experiments not marked complete here do not become implemented merely through documentation. Machine-readable sources: [PATHFINDER reference](paper_reference.json), [BANIS+ reference](baseline_reference.json).

## Establish what is being compared

| Work | Input and objective | Original comparison | Role in this track |
|---|---|---|---|
| PATHFINDER | Mouse EM → neuron reconstruction | FFN stages and full PATHFINDER on the same mouse volume | Method reproduction target; primary baseline should be FFN + traditional agglomeration |
| MaleCNS | Fruit-fly CNS EM → proofread connectome | Release pipeline combining automatic segmentation, synapse prediction, assistive tools, and manual proofreading | Accessible real fruit-fly training/evaluation data, not a single-model score |
| NISB / PyTorch Connectomics BANIS+ | Synthetic EM → instance segmentation | Fixed train/val/test, public checkpoint and nERL | Supplementary strong baseline/runtime check, not the PATHFINDER paper baseline |
| AxonEM-M | Mouse EM → axon reconstruction | Author training data and challenge evaluation; only partial references obtained locally | Completed training/engineering validation; potential OOD data requires split review |

Retain these distinctions:

- Reconstructing a connectivity graph is different from simulating neural activity or training fruit-fly behavior.
- Public methods/code do not establish that weights, original data, splits, and scoring materials for a paper experiment have all been obtained.
- FFN is a strong public method. BANIS+ was previously selected for its more complete weights/reproduction entrypoints, which does not establish suitability to replace FFN as the primary control.
- Proofread connectomes can serve as references but are not the output of one automatic inference run; labels are not infallible.
- Metrics called nERL are directly comparable only after aligning data, reference skeletons, and scoring rules.

## Two main experiments: FIB-25 companion data and MaleCNS

On 2026-09-28, the user approved adding an experiment with FIB-25 companion materials alongside the MaleCNS in-domain experiment. This section updates the earlier MaleCNS-only design below; OOD remains a later extension.

### What are FFN's original splits?

The [official FFN README](https://github.com/google/ffn) trains the public model on `validation_sample` (called `validation1` in training commands) and demonstrates inference on `training_sample2`. It provides no independent third model-selection block in this example. Directory names cannot establish training roles, and the example is not the complete paper benchmark.

These names come from [NeuroProof data](https://github.com/janelia-flyem/neuroproof_examples): the original workflow trains a boundary classifier on sample 1, a merge classifier on sample 2, and tests merging on validation. FFN uses them differently. Do not use NeuroProof weights trained on the target test block in comparisons.

The [2016 FFN paper §3.3, §4.1](https://arxiv.org/html/1611.00421) uses zebra finch, not FIB-25: 33 densely annotated training blocks (29 at 150³, 4 at 256×256×128), plus a skeleton evaluation volume. The skeleton volume also monitors checkpoints and stops training; it cannot be described unilaterally as a completely independent final blind test. Skeleton volume sizes in §4.1 and §4.3 differ and need original-material verification. The 2018 Nature Methods work was separately checked against the [author PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/4536.pdf) directly linked by the [Google paper page](https://research.google/pubs/high-precision-automated-reconstruction-of-neurons-with-flood-filling-networks/), rather than replaced with the 2016 version: J0126 zebra finch; model selection on 560×560×250 voxels and 221 skeleton fragments, minimizing merges then maximizing ERL; pipeline tuning on 12 neurites/13.5mm; final testing on 50 neurons/97mm, with nonoverlapping skeleton sets. This does not establish complete separation of training tissue and test neurons. The article states 35 dense training subblocks while Methods states 33; preserve the internal difference rather than invent unified membership.

That PDF also specifies distributed inference with 500³ windows,436 stride,64-voxel overlap; trim 32-voxel edges and recompute connected components, then stitch using mutual maximum overlap on overlap midplanes. Current independent 64³ tiles lack this stitching; larger tiles or consensus alone do not supply it. Local conservative stitching still needs independent implementation/verification without GT joins. The PDF is hash-pinned in Xtemp; see [source records](receipts/ffn2018-protocol-source.json). Current streamed tiled consensus requires first-input labels not to cross tiles; this condition cannot be reused after stitching without separate global-consensus verification.

`overlap_stitching.py` currently implements only shared-coordinate overlap-midplane extraction and mutual maximum-overlap matching; 7 synthetic tests passed, and it is not integrated into current formal results. A complete stitching entrypoint was not found in public pinned source, so this kernel is independent rather than a call to author stitching code. Explicit local choices: reject ties, allow background to compete for maxima without merging it, and select the lower middle plane for even overlaps. An interface projection kernel was added: when a plane is one voxel outside a core, map only face-adjacent voxels with the same original label into core components, preventing old IDs from reconnecting components separated by trimming. This is an explicit local boundary convention. `stitch_ffn.py` integrates official six-neighbor components and global assembly, checking strict window coverage/provenance/physical coordinates; 10 assembly/matching tests passed, including recovery of two separate objects from eight synthetic windows. `infer_overlap.py worker` reuses original FFN growth/model for overlapping windows; `merge` strictly aggregates shards before invoking `stitch_ffn.py`;3 shard-aggregation tests passed. Real Slurm GPU integration and real-data quality validation remain pending. A matching kernel alone does not complete paper distributed inference.

`launch_overlap.py` provides four-GPU scheduling: one independent worker per visible GPU, sharded by window ID; before launch it checks combined disk caps for all window outputs. A final manifest is written only after all four shards complete with matching provenance. Default 436³ cores plus 32 voxels per side form 500³ windows; models are not reduced. Jobs must already be registered as four-GPU `gpu` allocations, with 10 minutes reserved beyond inference. Assembly/evaluation are separate; inference completion does not establish quality. GPU isolation, subprocess error propagation, and shard completeness have unit tests; real Slurm execution remains pending.

Later worker versions additionally write `heartbeat.jsonl`: after model batches, at most every 60 seconds, record cumulative batches,33³ model calls, completed large windows, active window ID, and start/end states. Original `progress.jsonl` still records only persisted windows. Heartbeats explicitly set `quality_metric=false` and are not nERL-improvement or completeness receipts; long seed-preparation/model operations are not interrupted. Scheduler comparison tests confirm progress callbacks leave request ordering, predictions, and completion statistics unchanged. Existing running frozen snapshots lack this new recording; routine human/agent queries remain 30–60 minutes apart.

Full SHAPE comparison requires 625 prediction-context voxels (10µm) on each side of a 512³ scoring region, giving 1762³;1308³ experiments only diagnose FFN stitching. `infer_overlap.py worker` and `launch_overlap.py` provide explicit `--clip-boundary-to-image`, clipping only the outermost overlap envelope within the actual raw 16-voxel network halo, without trimming ROI or padding EM. This records `boundary_image_shape`; assembly verifies image grids and shards cannot mix boundary policies. Geometry checks passed for 125 windows covering the complete 1762³ ROI in 1794³ raw data, as did synthetic official-component/stitching checks after envelope clipping. This is an explicit local boundary convention, not claimed identical author code. Frozen 1308³ diagnostics do not enable this option.

To bound resident disk for full volumes, workers/four-GPU launchers offer `--compressed-window-bytes N` (default 0 saves original NPY). Positive values enable lossless per-window NPZ with a hard cap: write a compressed stream, verify the complete decompressed NPY hash before publishing, and fail on insufficient capacity without deleting labels or truncating predictions. The controller reserves the sum of all window caps. Stitching reads/verifies compressed windows on demand without full temporary extracted files; records retain both compressed-file and original-NPY hashes. Synthetic compressed/uncompressed inputs stitch voxel-identically. A real 256³ prediction shrank from 67,108,992 to 3,872,592 bytes; write plus decompression verification took 0.888 seconds, repeat verified reading 0.340 seconds, and peak process RSS 247MiB. This is not a 500³ compression-rate or full-volume peak estimate; reservations still use hard caps. See implementation_validations in the [six-run receipt](receipts/malecns-six-run-validation.json). Running frozen diagnostics do not use this option.

`pack_stitched.py --prediction --manifest --root --output --max-bytes --memory-gib` packs completed full stitched arrays into similarly verifiable lossless packages, retaining original NPY hashes, complete source receipt/manifest hashes, and coordinate grids. It checks completion/resource caps first and verifies decompressed bytes afterward, **without deleting original predictions**. This supplies inputs for one-at-a-time global consensus; only synthetic end-to-end validation is complete, and full-volume packing/six-run scheduling still require integration.

`consensus_stitched.py --spec --root --output --min-size --memory-gib [--slab-depth --max-pairs]` accepts an ordered JSON list of six package directories, verifying packages, parent stitching receipts/manifests, compressed bytes, original NPY identity, shared image/grid, and ascending/descending seed orders for each of three checkpoint-file hashes. It decodes one package at a time and uses output memmaps for global intersection. Memory estimates include decoding buffers, output pages, and global count tables; output does not depend on window-label ranges. Six synthetic variants passed actual official components, stitching, packing, and this entrypoint, matching official whole-array consensus partitions at two minimum-fragment thresholds. Duplicate variants, corruption, and memory/count-table overflows reject completion. This validates a CPU entrypoint, not a full real-volume experiment or paper checkpoint-selection protocol. The global intersection kernel also passed whole-voxel zero-mask/global bidirectional-label-mapping agreement against direct official consensus on the original six real 256³ predictions: 238,256 positive-label components,15.136 seconds for five intersections, peak RSS approximately 603MiB. This did not execute six real overlapping-window runs or establish quality improvement; hashes/resources are in the same receipt.

`prepare_prediction_context.py --prediction --image-manifest --root --output --memory-gib [--manifest --slab-depth --max-instances]` adapts full stitched/global-consensus outputs to existing SENSE/SHAPE contracts. Single stitched runs require original manifests; consensus checks six-package provenance. It retains parent receipts/labels and adds hard links, missing-image-layer records, complete instance indices, and one whole-ROI computation-coverage record, **without fabricating legacy independent-tile ID spaces**. Hard links still count full array sizes toward 200GB, so run only when space permits. Raw origins missing from old receipts can be derived only from inference-pinned image manifests, with physical-coordinate verification. Instance centers use hierarchical search preserving original coordinates/tie order and explicit instance caps; memory gates count array pages, hierarchical coordinates, and indices. Synthetic stitching/six-run consensus passed existing SENSE/SHAPE geometry checks and independent coverage audits; real full-volume downstream experiments remain pending. Complete computational coverage does not establish 10µm context for every instance or complete candidate coverage.

`score_stitching.py --manifest --stitched --suite --root --output` compares the same window predictions before/after stitching. Construct the unstitched scoring core from trimmed globally unique components, extract the matching stitched core, retain label IDs, and use the same frozen suite. Check equal foreground coverage, input hashes, and complete scoring-region coverage. It reuses the pinned ERL scorer; synthetic end-to-end tests actually called official components/scoring. Historical 64³ window results are contextual comparisons only, since simultaneously changing windows/stitching cannot isolate stitching gains. Real paired results remain pending.

Larger stitched instances increase SHAPE preprocessing memory. Development `indexed_mesh` computes centroids with integer axial counts, without retaining 3D coordinates for each foreground voxel; marching-cubes inputs/vertex/face ordering remain unchanged. `infer_shape.py --mesh-cache-bytes` bounds retained mesh cache to 512MiB by default, or 0 to disable it. Current motifs and extraction/sampling temporary buffers are outside that cap and need separate job-memory accounting. Eviction recomputes identical meshes without dropping candidates or reducing 8192 points. Tests match original meshes/centroids and complete 8192-point inputs across cache sizes. Full bbox masks, marching cubes, and sampling remain nonstreaming; cache caps are not process-memory caps. Existing frozen packages are unchanged. A future-experiment package passed CPU export/extension loading/identity audits: all three weight-file hashes and 372 model tensors match the original exactly, threshold remains 0.8; package manifest hash `81f79e7f5e2b40f416d29d691aae30fb1c04fca557a6c10d143f90f31dfdaf50`. New GPU forward/full-reconstruction validation is pending; it cannot replace historical runtime versions. See future_runtime_package_identity in the six-run receipt.

### Local execution protocol

| Experiment | Training | Validation/selection | Final evaluation |
|---|---|---|---|
| A0: official FFN example check | Author public model; disclose validation 1 exposure | No tuning; retain author settings | training_sample2; example reproduction only |
| A1: complete FIB-25 method comparison (not executed this round) | Small blocks cannot support full 10µm SHAPE radius | Do not reduce models to bypass this limit | B supplies the complete method comparison |
| B: MaleCNS in-domain comparison | Frozen r1 training region; exclude 555 public IDs shared with holdouts | Select checkpoints on frozen r2 spatial validation region | Frozen r3 test region; inspected pilots excluded from new comparison |

A0 approximately reproduced the official checkpoint example: VI 1.817109/1.817057, adapted Rand error 0.157358/0.157366 versus released output, and 0.0030848% differently labeled voxels. This is not byte-identical or paper-scalar reproduction and does not validate separately trained local FFN 15. See [official example results](receipts/fib25-official-example.json).

B uses [frozen splits](malecns-spatial-comparison.json). Data/full SHAPE inputs were prepared; training and corrected complete validation/test comparisons finished, with the learned method 3.44%/2.28% below the traditional baseline. Compare FFN-only, traditional FFN agglomeration sharing initial segmentation, and SENSE+SHAPE+search sharing initial segmentation; select models/configurations on validation only. Report per-region ERL/nERL, split/merge/omission, coverage, compute/disk. Complete 512³ cores are primary; eight prefrozen octants are correlated descriptive regional reports. Local skeleton metrics cannot directly be subtracted from paper numbers on other datasets.

Provenance of bucket object `tstvol-520-1` is unconfirmed; sharing a bucket does not make it FIB-25's third split. All data remains in Xtemp at≤200GB resident storage, only gpu and at most 4 GPUs per job; historical zero-shot plans do not launch automatically.

## PATHFINDER: training, controls, and scoring

Source: *Accelerating Neuron Reconstruction with PATHFINDER*, bioRxiv v1, 2025-05-21, [DOI](https://doi.org/10.1101/2025.05.16.654254). Actually checked [author full-text mirror](https://www.researchgate.net/publication/391957338_Accelerating_Neuron_Reconstruction_with_PATHFINDER), Table 1, Figure 4, and supplementary FFN training / SENSE / SHAPE / Segmentation evaluation. Original PDF bytes have not been obtained/hash-pinned; complete transcribed parameters and unresolved items are in `paper_reference.json`.

The team collected IBEAM-mSEM mouse cortex, originally 8nm isotropic; evaluation volume approximately 80.4×84.8×99.2µm³. References are manually proofread; primary quality evaluation covers all unmyelinated axons. This paper is not a cross-species “non-fly training→fly testing” experiment.

| Stage | Training material | Model selection/application |
|---|---|---|
| FFN v1.5 | 8.8×8.8×26.4µm³ densely annotated subvolume,16nm inputs; 20 residual modules, LayerNorm, AdamW | Select checkpoints on a 12.3³µm³ skeleton-annotated subvolume, then full-volume inference and oversegmentation consensus |
| SENSE | 1,684 manually reconstructed axons; local prompt→expanded segmentation of the same axon | 320 validation axons; 7 checkpoints selected from 3 field-of-view/resolution variants |
| SHAPE | 1,795 axon fragments; correct morphology as positives and incorrect SENSE combinations as negatives | 263 validation axons; F1 selection; 10µm radius,8,192 points, sampling/rotation ensembles at inference |

Training/validation descriptions do not establish obtained membership lists or strict spatial holdouts for final full-volume evaluation. Do not invent neuron-disjoint or OOD claims.

| Table 1 | nERL | splits | merges |
|---|---:|---:|---:|
| FFN v1.0 single | 0.685 | 44,417 | 1,282 |
| FFN v1.0 base | 0.562 | 79,230 | 200 |
| FFN v1.0 agglo | 0.766 | 33,835 | 347 |
| FFN v1.5 single | 0.785 | 28,207 | 196 |
| FFN v1.5 base | 0.661 | 50,981 | 20 |
| FFN v1.5 agglo | 0.820 | 23,085 | 33 |
| PATHFINDER | 0.942 | 2,862 | 702 |

`base` includes conservative consensus; `agglo` means traditional FFN merging. Our single FFN output is not directly paper base. The full method trades fewer splits for more incorrect merges; it does not outperform FFN agglo on every metric.

A 2026-10-01 recheck of that mirror's “FFN segmentation” supplementary Methods explicitly specifies three checkpoints, each run in forward/reverse seed order, with consensus over six segmentations. Traditional v1.5 merging requires all three checkpoint scores to meet thresholds. Checkpoints are selected by validation-segmentation edge accuracy; local top-three BCE checkpoints are not equivalent. The paper also uses initial vessel/glia segmentation and semantic-class processing, absent locally. These are explicit differences from current single-checkpoint controls.

Fragment filters cannot transfer across stages: 10,000 voxels applies to glial three-run consensus with semantic filtering; a generic neuronal six-run minimum-fragment threshold is not explicit in currently readable paragraphs. Cell bodies, nuclei, and myelinated axons also have class-specific consensus exemptions. Local `min_size=0` is a diagnostic configuration, not a confirmed paper parameter. Formal runs must explicitly freeze it and share the same filtered initial segmentation between traditional and learned methods. A10,000 threshold mentioned in the paper does not justify applying it to all fly neurons.

Merge thresholds also differ: the conservative Google FFN manual starting rule requires IoU>0.8, all four consistency values>0.6, and deleted/predicted<0.02 on at least one side. The PATHFINDER FFN v1.5 paper rule is `((min(deleted/predicted)<0.02) AND IoU>0.8 AND min(consistency)>0.8) OR (IoU>0.9 AND min(consistency)>0.9)`, with all three checkpoints passing. Current frozen single-checkpoint experiments use the former, rather than an exact paper baseline.

`merge_ffn_agglomeration.py --rule pathfinder2025` computes paper gates on complete candidate scores. `--ensemble --rule pathfinder2025` requires three distinct checkpoints with matching image/initial segmentation/candidate set, intersecting candidate passes across all three before global merging. Default `manual` preserves frozen experiments. 17 targeted tests passed; actual three-checkpoint end-to-end quality validation remains incomplete. Single-checkpoint validation threshold sensitivity is separate, without changing test configuration or establishing full paper reproduction.

Local `batched_ffn.py --seed-order ascending/descending` supplies an order-control entrypoint; default remains ascending. Resume/assembly reject mixed orders; continuous-run resharding currently supports ascending only. Six-run consensus on a 256³ validation subregion completed; three-checkpoint merging remains incomplete.

`consensus_ffn.py` directly calls pinned Google functions as a CPU entrypoint, requiring explicit input ordering, `--min-size`, and `--memory-gib`. It accepts complete single-shard predictions or full assembly with source receipts, checking array hashes, tile coverage, and label ranges. Different checkpoints/orders are allowed, but image, ROI, and coordinate grids must match. Official split consensus executes pairwise in input order and does not replace stitching.

Complete six-run inference must use `consensus_ffn.py --require-six-runs`. This rejects missing/duplicate checkpoint/order pairs, requiring ascending and descending for each of three checkpoint hashes. Callers still explicitly supply and retain ordering because consensus is noncommutative. This does not prove distinct weight tensors or selection by paper validation edge accuracy; both need independent evidence. Local three-checkpoint weights passed tensor-difference audits but are still BCE-selected.

`--tiled` uses the verified condition that initial labels do not cross tiles: all counts for an intersection lie in the same tile, so official per-tile calls preserve global segmentation equivalence up to positive integer relabeling. 150 synthetic comparisons against whole-volume official calls cover 2–6 inputs,64-bit labels, and small-fragment thresholds; real-volume resource/quality evidence remains missing. Both modes preflight memory/disk. `--output-dtype uint32` reduces output space but fails without completion receipts on label-count overflow.

`archived_consensus.py` provides streaming, verified equivalent on the same real 256³ validation subregion. `--spec` is an ordered six-item JSON list, each containing `prediction` (original assembly directory), `archive_receipt` (complete lossless archive record), and `group` (archive group containing only that segmentation.npy). Also require `--root`, `--output`, `--ffn-source`, `--connectomics-source`, `--min-size`, and `--memory-gib`. Six complete predictions must share images/grids/tiles. It reads NPY slices from zstd tar without restoring six arrays to disk, invoking official functions per tile. Completion requires verified compressed bytes, full decompressed NPY hashes, per-tile label ranges, and scientific-code hashes. Synthetic archives match nonempty official whole-array partitions and reject corruption/truncation/insufficient memory. 256³ output is byte-identical to direct consensus (CPU job approximately 4 minutes 26 seconds); no run result yet covers predictions for the full 1794³ context. It does not change inference, stitch across tiles, or supply missing semantic/checkpoint-selection protocols.

Stitched labels can cross windows, invalidating tiled consensus above. `global_consensus.py` adds a two-pass uint32 global intersection: first count whole-volume intersection sizes, then generate compact labels at the threshold, preserving official asymmetric background rules. Only positive label numbers differ from official code; retaining old IDs for maximum-overlap fragments does not change partitions. Actual pinned official functions were compared on 30 random cases×4 thresholds, sequentially processing 2–6 inputs and checking cross-slab counts, disk memmaps, and input preservation before resource-limit rejection. This kernel integrates with six-package compressed provenance validation, but real six-run Slurm execution remains incomplete and does not establish completed real global consensus.

Paper scoring includes TEASAR,100nm endpoint erosion, approximately 300nm node spacing, and split/merge node-support rules. Resolution mapping for 144-pixel boundaries, exact table search configuration, and complete inputs/splits remain unverified. Original-paper numerical comparability is therefore absent; “not obtained” does not assert “authors did not publish.”

## MaleCNS: a release pipeline, not PATHFINDER's fly test

Final version: *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell(2026), [DOI](https://doi.org/10.1016/j.cell.2026.08.015). **Methods read section by section in this audit come from [PMC preprint v2 (2025-10-30)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)**. Details below use that version; itemwise final-version differences remain unchecked. Data uses v1.0 from [official MaleCNS downloads](https://male-cns.janelia.org/download/).

Methods locations: EM Volume Segmentation / Synapse Identification / Proofreading / AutoProof.

| Stage | Paper description |
|---|---|
| Initial segmentation | FFN using MANC/optic-lobe reconstruction methods; hemispheres and ventral nerve cord processed separately, then stitched |
| Semantic classification/merging | Distinguish nuclei/glia etc.; match overlapping regions with constraints preventing incorrect joins |
| Synapse identification | Separate presynaptic/postsynaptic networks, fine-tuned on new annotations from this fly specimen with separate validation |
| AutoProof | Learn human focused-merge decisions to automatically join high-confidence isolated fragments |
| Final release | Extensive manual/semiautomatic proofreading; preprint estimates 44 person-years |

This cannot be simplified to “complete PATHFINDER generates MaleCNS zero-shot.” Exact production FFN weights, training-volume members, and all stage versions have not been obtained; generic FFN code is not substitute evidence. Google's [release blog](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) discusses FFN, PATHFINDER, and synthetic-neuron research without establishing use of all of them in this release. AutoProof and SHAPE are not the same model merely because both decide merges.

Our fly reference is locally skeletonized from official 16nm segmentation, an independent local protocol rather than author connectome quality metrics. Existing mouse-weight→MaleCNS pilots are additional local zero-shot transfers, rather than original settings of either paper.

## NISB / BANIS+: supplementary-control protocol

Sources: [official NISB benchmark/leaderboard](https://structuralneurobiologylab.github.io/nisb/), [pinned PyTorch Connectomics tutorial](https://github.com/PytorchConnectomics/pytorch_connectomics/blob/53fa46bf25d35966fd4d36a270ce84caf9c631b2/tutorials/neuron_nisb/README.md). This is a data benchmark plus external implementation, rather than part of the PATHFINDER paper.

| Item | Pinned setting |
|---|---|
| Data | Procedural neuron morphology + conditional-diffusion EM; base appearance source zebra finch j0126 |
| Splits | train seed0–4; val seed100; test seed101. Full training history not independently audited |
| Test | 3000×3000×1350,9×9×20nm; 416 reference skeletons; obtained images/skeletons approximately 12.34GB |
| Model | BANIS+ seed42, MedNeXt-L/k3, 61,779,399 parameters; released raw weights, no separately stored EMA |
| Inference | 128³ windows,50% overlap; author single-GPU batch 8; local four-GPU batch 2 is an acceleration variant |
| Decoding | Short-range affinity channels, numba connected components, threshold 0.66, edge_offset=0 |
| Threshold source | Author validation selection; retained locally without test threshold sweeps |
| Masks | Main comparison omits test-specific border/background masks |
| Reference numbers | Released-checkpoint guidance: mask-free approximately 0.596, specific mask approximately 0.601; tutorial 59.6±1.1% explicitly means mean±standard deviation across three runs |
| Metrics | Report-era em_erl merge_threshold=1; upstream changed default to 50 on 2026-08-20. Public-number checks must explicitly use 1, without mixing the strict local funlib protocol |

Distinguish native reproduction from transfer evaluation. Native test scores cannot be subtracted from fly nERL. Existing four-GPU 0.6296958 and single-GPU 0.6320541 use new merge threshold 50 and mismatch published protocols. Correction job 90000068 with original 1 completed: nERL **0.5883956420**, gap−0.0076043580 from approximately 0.596, passing the original±0.03 engineering check. The same predictions at 50 score **0.6296958174**, matching prior four-GPU results; protocol difference 0.0413001754. Four-GPU time 5545 seconds (6.1611 GPU-hours). This supports approximate released-checkpoint numerical reproduction, without exact agreement or retraining reproduction; the investigation range is not a confidence interval. See [original-protocol recheck](receipts/nisb-report-era-result.json); retain historical control in [early native results](receipts/baseline-gpu-profile.json).

## AxonEM: limits of existing engineering evidence

Sources: [author challenge page](https://axonem.grand-challenge.org/), [author project page](https://connectomics-bazaar.github.io/proj/AxonEM/index.html). Local provenance audit: [DATA_EXPANSION.md](DATA_EXPANSION.md).

Public mouse training blocks and continuous raw images have been used; obtained sparse references are not complete author challenge GT. Original Z sampling differs from the 16nm working grid; interpolation cannot restore missing detail. Historical evaluation risks exposure to training supervision and is not evidence of leakage-free tests or SOTA ranking. Future AxonEM OOD targets require explicit review of source-model training/target-test overlap; weights already trained on AxonEM cannot relabel the same tissue OOD.

## Local experiment changes, separate from paper facts

Direction proposed/discussed by the user on 2026-09-28: fly in-distribution comparison first, then independent-data OOD. Current execution status: **Corrected validation/test comparisons completed and passed execution-integrity acceptance; learned-method superiority remains unproven. Real six-run consensus and streamed-reading equivalence on a 256³ validation subregion completed; complete stitched multirun execution remains unverified.** `malecns-multiregion.json` retains the original zero-shot transfer design; current roles follow explicit `malecns-spatial-comparison.json`, rather than old transfer execution plans.

| Experiment | Training and selection | Test | Supported conclusion |
|---|---|---|---|
| Main: fly in-domain | MaleCNS training region; independent validation selects checkpoints/thresholds | Frozen spatial holdout | Method gains on the same dataset |
| Extension: cross-data OOD | Freeze main model and all thresholds | Independent EM, prioritizing review of available mouse cortex | Cross-data/species transfer, not the paper's original OOD experiment |
| Original-paper data integration | Separately verify author splits; retrain if necessary | Author ROI, axon members, scoring | Paper-number gaps only after complete alignment |

Main method groups:

1. FFN-only: initial-segmentation ablation.
2. **Same initial FFN segmentation + traditional FFN agglomeration: primary baseline.**
3. **Same initial FFN segmentation + SENSE + SHAPE + combination search: experimental method.**
4. BANIS+: supplementary strong baseline; separately list matched-training-set retraining and public-weight zero-shot use.

Arbitrary threshold merges cannot represent traditional agglomeration. Pin author implementation/algorithm, training/inference settings, and deviations. Reproducing paper oversegmentation consensus also requires separate multicheckpoint/seed-order cost records.

Splits precede crops: isolate spatial contexts and check neuron identity/supervision across regions. Spatial holdouts within one specimen are not automatically strict IID; shared neurons require explicit reporting. Inspected pilots/historical tests cannot become blind tests again; do not replace regions based on scores.

Use common physical coordinates/reference skeletons, without interpolating prediction labels. Methods may retain native grids but must report input resampling; converting 16nm fly inputs to 9×9×20nm does not restore native 9nm detail. Report per-region ERL/nERL, splits, merges, omissions, reference coverage, and training/inference costs. Few/shared-neuron regions warrant descriptive statistics, not invented independent-sample confidence intervals.

## Maintenance and the meaning of direct replacement

Before method changes or comparison reports, check data version/ROI/membership, training exposure, physical grids, model/checkpoint, validation selection, decoder thresholds, reference-skeleton processing, scorer version, manual involvement, and resource budget. Version changes as separate runs/protocols, preserving old results/failures; do not retrospectively change frozen tests to pursue paper scores.

“Direct replacement” after original-data release means reuse prepare/train/infer/evaluate interfaces with new adapters/manifests, rather than reuse old checkpoints to claim paper reproduction or overwrite files. All large data/temporary files remain in Xtemp at≤200GB resident storage, only Slurm gpu and at most 4 GPUs per job. Currently authorized cross-job concurrency is 20 GPUs; verify the ledger each time. New training needs finite budgets and peak-space checks on the original ledger.

Current measured results: [task README](../../README.md); execution: [selected starter](../../environment/baseline/pathfinder-fast/README.md). This document records protocols rather than live job status.

### Six-run consensus validation record (2026-10-01)

Frozen r2 subregion `[656,912]^3`,64³ windows, forward/reverse inference for each BCE-selected checkpoint 6750/7250/5750. Four-A6000 inference/consensus took 14 minutes 40 seconds (0.978 GPU·hours). Single 6750 ascending nERL 0.03646949 versus six-run 0.00802471; merge errors 44→17, split errors 721→1078, omitted-node fraction unchanged 0.65967476. Consensus is more conservative without establishing improved final quality. min_size=0 is explicit integration configuration; selection does not reproduce paper edge-accuracy selection. Archived streaming/direct outputs are byte-identical. See [receipts/malecns-six-run-validation.json](receipts/malecns-six-run-validation.json).

### Official candidate generation: whole-volume watershed and tiled reduction (2026-10-01)

Random-input official comparisons found that independent tiled watershed can change labels, candidate pairs, distances, and coordinates even when local EDT halo exceeds maximum expansion distance. Finite distance gates do not establish tiled-watershed equivalence. Six tiled configurations failed across four random cases in job 90000129; this implementation is not an equivalent acceleration of the official method.

Corrected `decision_centroid.py` first calls pinned Google's `watershed_expand` on the whole input, then uses two-pass tiled scans for candidate tables. Each neighborhood anchor belongs to one tile, preserving repeated candidate points/counts among all seven official offsets. Pass 1 computes each fragment pair's centroid over all globally minimum-distance points; pass 2 selects the nearest, preserving official offset/ZYX scan order for ties. This avoids changing centroids by retaining one representative per tile. 8 targeted tests passed, including earlier failing random cases. Job 90000130 separately compares official functions on a real 256³ FFN crop; results await acceptance.

Whole-volume watershed still needs substantial memory and is not end-to-end streaming. The new function requires explicit `memory_gib` and estimated-working-set checks; estimates are not measured peaks. Full 1762³ estimates exceed current 128GiB native configuration, requiring an independently validated high-memory CPU stage or proven equivalent memory optimization. Cluster `cpu` provides high-memory CPU nodes; this does not establish full-volume completion. Production candidate entrypoints have not switched; the complete official workflow has not passed delivery acceptance.

Initialization dependencies/frozen scoring checks were fixed: job 90000127 generated 27-stage commands and model copies for real r2 images, three FFN packages, and the frozen suite; job 90000128 passed 7 official-tool/frozen-scoring checks. Independent verification of 3,872 file hashes accepted these CPU records. They do not establish GPU execution, complete reconstruction, or performance gains. Original suites/scoring are unchanged; historical report wording changes were checked by reconstructing original full-file byte hashes.
