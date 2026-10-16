# Optional downstream research utilities

The delivered track evaluates a frozen encoder with the
[fixed probes](../../environment/evaluation/README.md). Its current baseline and results are
in [baseline/starter](../../environment/baseline/starter/README.md).

This directory contains optional developer tools for original-window extraction,
AWF/LFMC decoder training, missing-observation handling, and external-LFMC imagery
acquisition. They are not copied into the agent workspace and do not define the
track score or its baseline. Decoder training must not replace frozen-probe
scoring in a track submission.

The original-window tools expect the published rslearn datasets, rather than the
cropped HDF5 inputs used by the frozen probes. The external-cohort tools require
operator-owned inputs and selection policy; the repository does not supply a
complete external-cohort release. Use each command's `--help` for its interface.
No experiment status or model-quality result is reported in this directory.
