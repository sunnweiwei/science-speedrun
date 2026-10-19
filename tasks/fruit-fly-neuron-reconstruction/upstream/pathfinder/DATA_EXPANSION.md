# Continuous-volume expansion survey (2026-09-26)

Execution update: the complete ZIP was downloaded and hash-verified; see [raw-image verification](receipts/full-raw-download.json). “Not downloaded” below refers to the original source-survey date.

Conclusion: **Complete continuous raw AxonEM-M EM is publicly obtainable, and 200 GB can hold raw images and one set of working arrays; a download entrypoint for complete public axon GT remains unconfirmed.** Enlarging raw images satisfies geometric context requirements for SHAPE's 10 μm radius, without automatically supplying morphology training samples, GT coverage, or paper comparability. This survey checked only author pages, JSON metadata, HEAD, and an approximately 192 KiB Range request for the ZIP directory; no large data download or GPU execution occurred.

## Public sources that can be rechecked

The [author challenge page](https://axonem.grand-challenge.org/) directly links raw images, nine training blocks, and a Neuroglancer viewer; viewer text explicitly says only approximately 1% of GT is displayed. The [author project page](https://connectomics-bazaar.github.io/proj/AxonEM/index.html) describes full data and reference work; statements about complete research GT do not establish that corresponding files have been obtained.

| Contents | Verified in this survey | Size/scope and purpose |
|---|---|---|
| [Pinned AxonEM-M raw images](https://huggingface.co/datasets/pytc/EM30/resolve/63519706548316c1f00863cbea0ece40358a759a/EM30-M-im-pad.zip) | HEAD 200; Content-Length **15,207,108,024**; Accept-Ranges=bytes; actual Range GET 206 | ZIP directory contains 790 PNGs (`im0000.png`…`im0789.png`) and 1 directory; extracted PNG files total **15,205,306,740 B**, rather than decoded-pixel memory |
| [Raw Neuroglancer info](https://rhoana.rc.fas.harvard.edu/ng/EM30-M/im_pad/info) | uint8; JPEG encoding; XYZ size **5120×5120×800**; original grid **8×8×40 nm** | Public chunks support random access; multiscales include 16×16×40, but JPEG pixels cannot be assumed losslessly identical to PNG |
| [Sparse-label info](https://rhoana.rc.fas.harvard.edu/ng/EM30-M/gt_axon_sparse/info) | uint32; highest public XYZ resolution **64×64×80 nm**; 512×512×375 | Physical extent **32.768×32.768×30 μm**; approximately 393 MB decoded labels, now downloaded; 68 nonzero IDs, not complete GT |
| [Pinned training repository](https://huggingface.co/datasets/pytc/AxonEM/tree/af20076475846ffa8a68d964a29fd4ca23eb66c2) | Linked by authors; public tree contains only human/mouse nine-block training archives and documentation | Mouse archive already used; complete evaluation GT not obtained from this repository |
| [Official evaluation code](https://github.com/PytorchConnectomics/AxonEM-challenge) | README references `axonM_gt_16nm_skel_stats.p`, absent from public tree | Public code does not establish public downloads of required GT skeletons; do not guess private object paths |

Pinned raw-image Hugging Face commit: `63519706548316c1f00863cbea0ece40358a759a`. LFS-declared SHA-256: `93824734ec7c2c09df87f8b01e32a556446c437988618b27b5234df76659eccf`; this is expected metadata from author storage. The complete file was not downloaded in this survey, so content hashing was incomplete. Repository metadata license: MIT. Complete ZIP central-directory SHA-256: `76fb69f158d80b90aa990a5a4b1da5e04d6cfaaeabf92baf84d1b9bda0bbee9c`.

Harvard info SHA-256 values: `781240078a2ec454aba76bca94d9dea80da6ab8878f0670c9eeb304b09530fda` (raw) and `9cac6a4d254b3974b8bf183baf16aa590073d35b0f48bc6efb9626e8ad0c86d4` (sparse). This public host's certificate was expired; only unauthenticated reads from that host used a certificate exception, consistent with existing sparse receipts. HF access used normal TLS verification.

## Coordinate and annotation issues to resolve first

- [Original paper §2.1](https://donglaiw.github.io/paper/2021_miccai_axonEM.pdf) gives mouse data **7×7×40 nm**; current challenge/viewer metadata use **8×8×40 nm**. Retain current release 8nm coordinates and explicitly record the paper difference; do not silently switch to 7nm and reuse existing meshes, centers, or physical model fields of view.
- Paper §2.2 distinguishes nine dense **512×512×50** training annotations from whole-volume evaluation of axons longer than 5μm. Training IDs do not automatically imply axon semantics. Author materials inspected here do not establish a uniform biological category for **label 1**; do not infer background or axon status.
- The challenge page specifies padding on both sides as ZYX `[20,512,512]`; 750+40=790 matches ZIP slice count, while the viewer declares 800 Z slices. Prefer the pinned ZIP and verify offsets by image correspondence; do not treat viewer 800 slices as ZIP 790 slices.
- The author's viewer adds XYZ offset `[512,512,20]` to sparse labels in raw `[8,8,40] nm` units, yielding `[4096,4096,800] nm`. With core origin zero, the raw outer-corner ZYX origin is `[-800,-4096,-4096] nm`. Sparse GT origin must not be set to the raw-image outer corner.
- The ZIP directory lists 11 PNGs smaller than 100KB, far below ordinary slices. **These are anomalies to inspect, not proven blank/missing slices.** Decode after downloading to check valid image coverage; nominal context must not conceal actual missing slices.

## Execution plan and estimates within 200 GB

Read complete lossless raw images or a fixed continuous ROI, verify alignment, and resample blockwise to a 16nm working grid. The complete 32.768×32.768×30μm core is approximately 2048×2048×1875 voxels: uint8 images **7.864 GB**, one uint32 prediction **31.457 GB**, and one boolean mask **7.864 GB**. ZIP approximately 15.207GB; retained PNGs approximately 15.205GB; assuming each decodes to 5120² uint8, 790 slices total approximately 20.709GB. **These are disk estimates, not verified array shapes/occupancy.** One core raw+prediction+mask totals approximately 47.2GB and stays below 200GB with source data; multiple logits, ensembles, or label copies may exceed the cap. Decode as a stream, avoid retaining all intermediate representations, and recalculate existing occupancy before writing.

A complete 20μm-diameter sphere is valid only when its center is at least 10μm inside volume boundaries. Cropping only a 20μm cube leaves essentially one valid center. Retain continuous predicted context larger than 20μm and evaluate an interior region separately. **Disk feasibility does not establish whole-volume segmentation within one day on four GPUs.** First measure representative continuous-block throughput, then freeze a budget-feasible prediction region; unpredicted regions are not completed context.

Sparse GT supports diagnostics only on annotated axons. Unannotated voxels are unknown, rather than negative labels, and cannot fully penalize incorrect merges between annotated and unannotated axons. Preserve existing spatial test splits; initially treat continuous-data work as development experiments. Final trusted long-range evaluation still needs complete author GT, official skeletons, or an independent explicit annotation protocol.

## Scope of alternative large-data surveys

| Author source | Relationship to this task | Current assessment |
|---|---|---|
| [MICrONS Layer 2/3 v185.1](https://www.microns-explorer.org/layer-23-v1851) | 250×140×90μm mouse visual cortex; 3.58×3.58×40nm; public dense segmentation and proofread axons/dendrites of 364 excitatory neurons; authors support local CloudVolume reads | Candidate for large context/morphology data, but anisotropic and not error-free GT for every axon; pin version/proofread membership and adapt separately. No download or specific ROI confirmed |
| [CellMap challenge](https://cellmapchallenge.janelia.org/page/about) | Author-provided 4 or 8nm isotropic eFIB-SEM, multispecies/multitissue cell/organelle annotations | Isotropic imaging is not long-axon instance GT. A subset jointly satisfying mouse cortex, >20μm continuity, and complete axon-instance annotations was not verified; not a ready replacement |
| [Janelia MaleCNS downloads](https://male-cns.janelia.org/download/) | Authors explicitly provide 8nm isotropic proofread neuron segmentation, EM, and skeletons, with block reads possible | A genuinely isotropic large morphology source, but fruit fly rather than mouse; does not replace the mouse morphology domain and can only form a separate transfer/engineering dataset |

No alternative dataset was confirmed to jointly satisfy “public, genuinely isotropic, continuous>20μm, mouse cortex, and complete companion axon GT.” Prioritize continuous AxonEM raw-image expansion for engineering context; do not claim it resolves GT or paper-data gaps.
