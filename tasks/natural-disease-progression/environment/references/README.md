# References (not verified in this setup)

- [`delphi/`](delphi/README.md) vendors the upstream Delphi training implementation and synthetic
  demo dataset at a pinned commit.
- [`scaffold/`](scaffold/README.md) is an architecture-neutral data, training and submission
  scaffold with explicit seams to implement; it is not a model.

The verified neural baseline is `/workspace/baseline/temporal-gru`. None of these constrain
sequence length, sampling, model dimensions, pooling, loss, optimizer, fold selection or
architecture. A final submission must make and document a structural change rather than merely
rerun a reference configuration.
