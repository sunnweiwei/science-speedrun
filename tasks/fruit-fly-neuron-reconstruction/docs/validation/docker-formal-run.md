# Formal verifier run under Docker (H100)

Run on 2026-10-17 to 2026-10-19 on an 8×H100 host, using four GPUs
(`device=0,1,2,3` or `0,2,3,7`), 64 CPUs and 512 GiB, with the Compose
security settings and `tests/test.sh`. Package: the Antigravity agent session
`ff-r1`, which followed the baseline recipe for FFN (legacy step 6750, then one
production update; FFN sha256 `ee5e9fd5…`) and trained SENSE for 1,500 steps and
SHAPE for 300 steps under a 30-minute cap.

| Stage | Wall time (4 GPUs) | Notes |
|---|---:|---|
| seeds | 588 s | 125 windows, exact PolicyPeaks cache |
| FFN | 17,151 s | 16 workers under MPS, `norm32`; 53.2 M calls; 20 windows at the 1 M call cap |
| stitch | 1,345 s | 298 M face-connected pieces before the 10-voxel minimum, 17.8 M after |
| context | 1,408 s | |
| SENSE | 4,015 s | 443,824 centers, 16 shards |
| merge, proposals | 33 s | |
| FFN-pair | 34 s | 16 shards |
| SHAPE | 32 s | 16 shards |
| finalize | 108 s | |
| scoring | 106 s | |

Total about 6.9 h on four GPUs. FFN and SENSE scale with GPUs; stitch and
context are single-process CPU stages. On eight GPUs this is about 4 h.

Reward `nerl` 2.214589e-05 (ERL 1.25 nm of 56.3 µm maximum; 4,103 merge and
4,389 split errors; 272,234 of 575,913 sampled nodes omitted). FFN-only,
shared-proposal FFN and disjoint-pair SHAPE scored the same.

Failures found and fixed on the way (ai4sci-tasks#150):

1. Shards that contained a window at the per-window call cap were marked
   incomplete, so no model with such a window could be scored.
2. 280 M pieces under 10 voxels exceeded the 60 M context label bound.
3. The namespace did not bind /etc/gitconfig, so the runtime checkouts failed
   git's ownership check in stitching.
4. The frozen suite pins the pre-rename common.py (#107 changed only the native
   Slurm partition name); scoring refused to run.

After acceptance, `reward.json` was reduced to numbers only (reward plus each
method's raw metrics, as Harbor requires); provenance stays in `comparison.json`.
Scoring was rerun on the same prediction with the same reward.

After fix 3 the run resumed from stitching in the same namespace on the existing
FFN output, and after fix 4 scoring ran on the finished prediction; FFN was not
repeated. Window outputs of the parallel launcher and `norm32` match the
earlier eager runs bit for bit on seven complete windows.
