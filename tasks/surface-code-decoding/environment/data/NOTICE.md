# Attribution and provenance

Experimental measurements, ideal circuit descriptions and published calibrated
detector error models derive from Google Quantum AI's Sycamore surface-code
memory experiments, Zenodo record 6804040:
https://zenodo.org/records/6804040

The source data is distributed under Creative Commons Attribution 4.0
International (CC-BY-4.0): https://creativecommons.org/licenses/by/4.0/
Changes: the benchmark selects surface-code d3/d5 experiments, partitions
source rows into train/validation/test, replaces row indices with stable opaque
identifiers, converts detector/label arrays to NumPy NPZ, and separates
inference inputs from grader labels. Original data provenance is preserved in
release metadata and content hashes. The release does not alter the experiment.

The scientific AlphaQubit reference is Bausch et al., “Learning high-accuracy
error decoding for quantum processors,” Nature 635, 834–840 (2024),
https://doi.org/10.1038/s41586-024-08148-8 . The bundled decoder implementation
and weights are independent benchmark reproductions, not official Google code
or weights. Code/dependency notices and licenses remain applicable; this dataset
card's CC-BY-4.0 metadata describes the experimental data and does not relicense
third-party libraries. Consult the linked code repository and retained upstream
notices for software licensing. No endorsement by Google Quantum AI or Google
DeepMind is implied.
