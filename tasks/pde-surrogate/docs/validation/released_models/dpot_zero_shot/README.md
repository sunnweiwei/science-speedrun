# Reference: released DPOT checkpoints, zero-shot on PDEArena Navier-Stokes

DPOT (Hao et al., ICML 2024) was pretrained on twelve 2D datasets including the
PDEArena Navier-Stokes **training** split, so its numbers on this task's
PDEArena held-out cases (test shards, unseen by DPOT) are in-distribution and
are reported with that caveat. Weights are Apache-2.0 (`hzk17/DPOT`, pinned);
the code repositories carry no license, so `predict.py` imports them from
`DPOT_CODE_DIR` at run time and the task ships none of it.

Two settings: `DPOT_CONTEXT=pad` (task protocol: four context frames, the
first repeated to fill DPOT's ten-frame window, K = 10) and
`DPOT_CONTEXT=native` (ten context frames, K = 4; a different protocol, for
context only).

```bash
DPOT_CODE_DIR=/path/to/DPOT DPOT_CKPT="$TASK_PRIVATE_DIR/pretrained/dpot/model_Ti.pth" bash infer.sh INPUT_DIR OUTPUT_DIR
```
