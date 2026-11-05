# Development data

This directory contains only the development view prepared by the controller.
It does not contain DUD-E or LIT-PCBA, raw downloads, benchmark checkpoints,
or test-related metadata.

    .complete.json
    processed/profsa_pretrain_500k/data.lmdb
    processed/biolip_train/data.lmdb
    processed/biolip_dev/data.lmdb
    processed/manifest.json
    splits/train.txt
    splits/dev.txt
    splits/profsa_pretrain_500k.txt
    pretrained/manifest.json
    pretrained/mol_pre_no_h_220816.pt
    pretrained/dict_mol.txt
    pretrained/dict_pkt.txt

LMDB uses a single file, contiguous ASCII integer keys starting at zero, and
pickled dictionaries. Paired BioLiP records contain atoms, coordinates (a
list of ligand conformers), pocket_atoms, pocket_coordinates, smi, pocket,
and complex_id. ProFSA additionally provides lig_atoms_real and lig_coord_real
with equivalent molecule aliases. Coordinates are in Ångströms. The pair's
label identifies a known training association; contrastive negatives are
constructed by the training method.

Use ProFSA and BioLiP train for learning, and BioLiP dev only for validation
and selection. The compact release samples 500,000 ProFSA records and
reconstructs BioLiP/Q-BioLiP; it is not the paper's exact training population.
The materializer applies the repository's fixed protein/ligand contamination
filters before creating these views.

The only supplied pretrained weights are the generic Uni-Mol molecule
initialization pinned in manifest.json. The reference validates sizes and
SHA256 hashes before use. Your exported candidate must own its final weights
and run without these initialization files.
