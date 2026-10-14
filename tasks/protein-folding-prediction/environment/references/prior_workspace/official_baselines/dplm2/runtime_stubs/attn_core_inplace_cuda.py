"""Import guard for an unrelated OpenFold extension in DPLM's eager imports.

DPLM-2 training does not call OpenFold's attention kernel: training consumes
pre-tokenized structures.  The upstream package nevertheless imports all model
families eagerly.  This guard makes that import possible and fails explicitly
if a code path actually tries to use the absent kernel.  Coordinate decoding
must use an image with the real extension, never this guard.
"""


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "attn_core_inplace_cuda is not installed in the training-only image; "
        "use the DPLM-2 decoding image for structure-token decoding"
    )


forward_ = _unavailable
backward_ = _unavailable
