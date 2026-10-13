# Upstream licenses and attribution

The task design, data semantics, metric definitions, and Fourier-feature
DeepSets starter are derived from:

> MassSpecGym: A benchmark for the discovery and identification of molecules
> Copyright (c) 2024 MassSpecGym developers
> https://github.com/pluskal-lab/MassSpecGym

MassSpecGym code and its Hugging Face dataset are released under the following
MIT license:

```text
MIT License

Copyright (c) 2024 MassSpecGym developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Git does not redistribute the large MassSpecGym source files. The task-owned
Hugging Face prepared release redistributes derived MassSpecGym views and the
MIT-licensed sanitized ICEBERG augmentation with this attribution.
Operators download the pinned upstream release directly and should retain its
dataset card, paper citation, and license with any derived data view.

The reference-baseline specification also describes the MIST and MIST-CF
projects:

> MIST: Metabolite Inference with Spectrum Transformers
> Copyright (c) 2022 Samuel Goldman
> https://github.com/samgoldman97/mist

> MIST-CF: Chemical Formula Inference from Tandem Mass Spectra
> Copyright (c) 2023 Sam Goldman
> https://github.com/samgoldman97/mist-cf

Both original repositories use the MIT license. The strengthened downstream
encoder is taken from the `MIST-FRIGID` branch of `coleygroup/FRIGID`. That
branch does not itself contain the top-level license files found on FRIGID's
main branch, so this task links to it but does not vendor it; clarify the branch
license before redistributing adapted source.

The reference data/model records have separate metadata:

- Zenodo `15008938` (exact v1.5 MIST processed data) is MIT;
- Zenodo `20671842` (filtered synthetic augmentation) is MIT; and
- Zenodo `19685145` (released FRIGID checkpoints) is CC BY-NC 4.0.

The prepared dataset redistributes only the MIT-licensed, sanitized synthetic
training data from Zenodo 20671842. No MIST, MIST-CF, or FRIGID source or released
checkpoint is bundled in that dataset. Preserve all upstream attribution and
license terms if a task-owned reference package later vendors source or model
artifacts. In particular, do not silently bundle the non-commercial checkpoint
archive into a generally redistributable benchmark image.

The GLACIER reference adapter uses the public `coleygroup/ms-pred` project:

> GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem
>
> Copyright (c) the ms-pred contributors
>
> https://github.com/coleygroup/ms-pred

The source repository is MIT licensed. The task pins and links to, but does not
redistribute, the separately hosted MassSpecGym GLACIER checkpoint. The
checkpoint download does not currently carry separate license metadata; obtain
clarification from the authors before redistributing the weight itself. The
reference preparation script records its exact byte count and SHA-256 so a
future license clarification cannot silently change the evaluated artifact.
