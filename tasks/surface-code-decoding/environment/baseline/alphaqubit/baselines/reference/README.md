# Pinned QRENND reference

`qrennd/` is the complete unchanged Python package from
[Hossein-D/qrennd](https://github.com/Hossein-D/qrennd) at commit
`c2ed73e3ec6ee843293c8839bf059c63b2960e61`. The original project is credited to
Boris M. Varbanov and identifies `BorisVarbanov/qrennd` as its repository. That
original address was unavailable during this reproduction; this is a preserved
mirror, not a newly authored implementation.

The original Apache-2.0 `LICENSE` and two historical Google experiment configs
are retained. `provenance.json` records SHA-256 hashes of every vendored source
file, config, and license. No upstream source files were modified. Generated
Python bytecode is not part of this copy.

The wrapper in `../neural.py` invokes native `Config` and `get_model` functions.
It supplies the published experimental-noise model widths (64 and 253), rather
than the historical d5 config's width of 128. The runtime uses a separately
pinned modern environment; upstream's original Python 3.8.12 / TensorFlow 2.4.1
installation metadata is not used.

See `../../reproduction/upstream-audit.md` for architecture, protocol
differences, and input adaptations. Source provenance verifies the implementation
copy; it does not imply reproduction of the paper's complete experimental
protocol or reported accuracy.
