# Available data

Read-only training data are under `/workspace/data/training/volume` and
`/workspace/data/training/shape`. Their manifests contain train and validation
records only. Use `/workspace/data/validation/image` and its visible `suite`
for development evaluation. All derived data belong under `/workspace/research`.

Raw image coordinates are ZYX, voxel-outer-corner, nanometres, on the recorded
16nm grid. Unknown tissue and excluded shared IDs are not negative labels.
Do not change voxel spacing by editing metadata. Spatial separation and shared-ID
exclusions do not establish complete biological independence.

`/workspace/data/runtime` supplies operator-pinned FFN/connectomics source and
FPS binary/source/licenses. It contains no model weights or predictions. The
test image and its reference suite are not in this workspace.
