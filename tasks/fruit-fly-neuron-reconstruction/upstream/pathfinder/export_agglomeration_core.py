"""Trusted evaluation-side export: global FFN merge map applied to scored core."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from merge_ffn_agglomeration import remap
from prediction_core import core_slices
from dataset_contract import sha256
from fetch_axonem import reserve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import write_json,external


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','base','base_metadata','mapping','suite','output'):
        p.add_argument('--'+name.replace('_','-'),required=True,type=Path)
    a=p.parse_args();out=external(a.output);root=external(a.root)
    if root not in out.parents or out.exists():raise ValueError('New output under original root required')
    mapping=json.loads(a.mapping.read_text());suite=json.loads((a.suite/'suite.json').read_text())
    if mapping['base_sha256']!=sha256(a.base):raise ValueError('Global merge map belongs to a different base')
    for name,digest in suite['files'].items():
        if sha256(a.suite/name)!=digest:raise ValueError('Reference suite changed')
    metadata=json.loads(a.base_metadata.read_text());reference=json.loads((a.suite/'metadata.json').read_text())
    base=np.load(a.base,mmap_mode='r');sl,grid=core_slices(metadata,reference);crop=base[sl]
    reserve(root,crop.nbytes+1_000_000);out.mkdir(parents=True)
    dest=np.lib.format.open_memmap(out/'segmentation.npy',mode='w+',dtype=crop.dtype,shape=crop.shape)
    table={int(k):v for k,v in mapping['mapping'].items()}
    for z in range(0,crop.shape[0],16):dest[z:z+16]=remap(crop[z:z+16],table)
    dest.flush();del dest
    write_json(out/'metadata.json',grid)
    write_json(out/'export-receipt.json',dict(global_mapping_sha256=sha256(a.mapping),
        base_sha256=mapping['base_sha256'],suite_sha256=sha256(a.suite/'suite.json'),
        candidate_evaluation_complete=mapping['complete'],missing_pairs=mapping['missing_pairs'],
        incomplete_pairs=mapping['incomplete_pairs'],accepted_pairs=mapping['accepted_pairs'],
        segmentation_sha256=sha256(out/'segmentation.npy'),metadata_sha256=sha256(out/'metadata.json'),
        policy='Global union before crop; exact label remapping, no interpolation or core-local union'))


if __name__=='__main__':main()
