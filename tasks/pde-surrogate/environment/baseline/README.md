# Baselines

Reference implementations verified in this setup. Each one runs in this workspace and publishes a
package that follows `/workspace/contract.md`. `RESULTS.md` in each directory holds the numbers
we reproduced with the independent verifier. No weights are shipped.

| baseline | what it is | reward |
| --- | --- | ---: |
| [`starter/`](starter/) | 473k-parameter channel-shared U-Net across every dataset; proves train → publish → replay | 0.3127 |
| [`cnext_unet/`](cnext_unet/) | The Well's CNextU-net recipe (18.6M) trained from scratch on PDEArena; one model per dataset | 0.3447 |

For scale: persistence scores 0.2827, the spatial mean of the last frame 0.3105, and the exact
continuation 1.0.
