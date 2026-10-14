# ESMFold2 official-architecture 50M baseline

The folding network is a 49,198,301-parameter configuration-only scale-down of
Biohub's released `ESMFold2Model`. The frozen ESMC-6B backbone is outside that
budget. No trainable layer is locally reimplemented. The unchanged pinned
checkpoint remains available through `inference.py` for official-reference
FoldBench inference.

Biohub's top-level forward is inference-only, although the release contains a
differentiable diffusion denoiser. `trainable_forward.py` executes the exact
released input/trunk/PARCAE/CODA path with gradients, stops before ODE sampling,
and calls that denoiser for an EDM training step. The loss, optimizer and
schedule are our `foldingtrain_edm_v1` reconstruction, not Biohub's unpublished
Recipe. The production runner provides deterministic distributed one-pass
sharding, exact precomputed ESMC mixed-256 inputs, EMA, atomic checkpoints,
per-rank RNG resume and common-view validation. `attach_cached_features.py`
joins only validated cache objects to the audited native split. The cached
boundary is the released `base_z_linear + base_z_combine` output; the remaining
`base_z_mlp`, trunk and structure modules continue to execute normally with
gradients. Online inference recomputes that same frozen boundary from ESMC-6B.

The production seven-GPU gate completed a native cached-feature forward,
diffusion+distogram loss, backward, deterministic gradient synchronization,
clipping, optimizer/EMA updates, checkpointing and held-out validation. The
confidence head is intentionally frozen in this training stage. This gate is a
training-system test, not a FoldBench score.
