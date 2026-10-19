# Validation scope

The track has completed native GPU training, reconstruction and scoring.
Report validation by execution path; absence of Docker on the author host is
an environment limitation, not a task-wide experimental designation.

| Component / execution path | Evidence and limits |
|---|---|
| Historical native full-volume comparison | FFN, SENSE and SHAPE completed; independent audits and a byte-identical downstream repeat passed. This is not the selected starter’s development baseline. |
| Current FFN development evaluation | Complete single-window and stitched eight-window GPU evaluations distinguish models. Selected starter nERL: proxy 0.0722669, single window 0.0707893, eight windows 0.0350171. Different scopes must not be compared as score improvements. |
| Training | Corrected-from-start training and its downstream FFN evaluations completed. The selected recipe is legacy step 6750 plus one production-rule update; an independent fresh replay of that entire selected recipe is not recorded. |
| Formal entry: `tests/test.sh` → `verify_single.py` | Ran end to end under Docker on four H100s (resumed after FFN once and before scoring once, after fixes); see [run record](validation/docker-formal-run.md). |
| Submission and scoring components | Tests cover data-only weight staging, package rejection, namespace construction, timeout cleanup and score/failure semantics. Real checkpoint export and relocation checks passed. |
| Isolation | Prediction ran on GPUs inside the bubblewrap namespace under Docker; the real-namespace test passes in the verifier image. |
| Docker / Harbor deployment | Docker verifier run on an 8×H100 host with the Compose namespace settings (seccomp/AppArmor unconfined, unmasked /proc, SYS_ADMIN). |
| FPS runtime | Prepared native runtime import and model checks passed within their recorded scope. Binary compatibility with a different target Python/Torch/CUDA/GPU stack requires a target check. |
| Resources | Measured formal run: about 6.9 h on four H100s (FFN 4 h 46 min, SENSE 67 min, stitch + context 45 min CPU), about 4 h projected on eight. Verifier timeout 12 h. |
| Public data distribution | Pinned 14.54GB / 722-file HF release; independent anonymous download and per-file hash checks passed. A fresh rebuild from upstream data is separate and remains unverified. |

## The formal run

The [Docker run record](validation/docker-formal-run.md) covers a fresh model
package and raw image passing through `tests/test.sh`, container checks, weight
staging, isolated prediction, FFN/context/SENSE/FFN-pair/SHAPE stages, private
scoring and `reward.json`. Two resumptions after fixes reused the finished FFN
output inside the same namespace; a single uninterrupted run is still to be
recorded with the next trained package.

`verify_single.py` requires the image-owned container marker and fixed scratch
path before inference. It cannot be run as an ordinary native CLI on the author
host. Do not bypass these checks to present a native experiment as a formal
verifier run. A deployment run should preserve the wrapper and record its
status, model/input hashes, reward and complete resource measurements.

## How to interpret quality and cost

The old full-volume table measures a different scope from current development
comparisons. Its lack of SHAPE gain is not evidence that FFN training changes
are invisible to the metric. Conversely, proxy performance does not establish
full-volume performance. Coverage and transfer from proxy to stitched regions
remain scientific limitations.

Eight-window timings include a 90-minute four-GPU run and a 4h31m50s four-GPU
run for different checkpoints. Do not replace the historical full-volume cost
with a universal two-hour estimate. Use smaller development evaluations for
screening, followed by larger-region confirmation.

## Evidence

- [Current results and cost scopes](../README.md#results-and-validation)
- [Detailed starter comparisons](../environment/baseline/pathfinder-fast/RESULTS.md)
- [Completed eight-window evaluation](validation/random-start-multiblock.json)
- [Completed corrected-from-start training and evaluation](validation/random-start-final.json)
- [Formal-entry component checks](validation/formal-route-engineering.json)
- [Public resource release](validation/huggingface-release.json)

Dated receipts retain their original scope and claims; later evidence may
supersede their pending items. They are maintainer records, not agent commands,
and are excluded from the development image. Start with the task README,
contract, materializer and tests/README for current operation.
