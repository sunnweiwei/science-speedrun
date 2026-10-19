"""Verify and assemble independent FFN shards, preserving unfinished coverage."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from batched_ffn import grid_tiles
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from fetch_axonem import reserve


def validate_shards(receipts):
    if not receipts:raise ValueError('At least one shard receipt required')
    first=receipts[0];seen_shards=set();tiles={}
    keys=['roi','tile_size','id_stride','shards','checkpoint_sha256','raw_sha256','metadata']
    start=np.asarray(first['roi'][::2]);stop=np.asarray(first['roi'][1::2])
    expected={t['index']:t for t in grid_tiles(start,stop,first['tile_size'])}
    for receipt in receipts:
        if receipt.get('ground_truth_used') is not False:raise ValueError('Prediction-only provenance required')
        if receipt.get('seed_order','ascending')!=first.get('seed_order','ascending'):raise ValueError('Mixed seed orders are not shards of one inference')
        if any(receipt[k]!=first[k] for k in keys):raise ValueError('Incompatible shard provenance or geometry')
        shard=receipt['shard_index']
        if type(shard)!=int or shard in seen_shards or not 0<=shard<first['shards']:raise ValueError('Duplicate/invalid shard')
        seen_shards.add(shard)
        for tile in receipt['tile_results']:
            index=tile['index']
            if index in tiles or index not in expected or index%first['shards']!=shard:raise ValueError('Duplicate/foreign tile')
            if any(tile[k]!=expected[index][k] for k in ('start','stop')):raise ValueError('Tile geometry mismatch')
            tiles[index]=tile
    return first,expected,tiles


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipts',type=Path,nargs='+',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image-manifest',type=Path);a=p.parse_args();root=external(a.root);out=external(a.output)
    if root!=out and root not in out.parents:raise ValueError('Output must be inside experiment budget root')
    receipts=[json.loads(p.read_text()) for p in a.receipts];first,expected,tiles=validate_shards(receipts)
    shape=tuple(first['metadata']['shape']);dtype=np.uint32 if len(expected)*first['id_stride']<2**32 else np.uint64
    reserve(root,int(np.prod(shape))*np.dtype(dtype).itemsize+10_000_000)
    # Validate every source BEFORE allocating the large assembled output.
    for index,tile in tiles.items():
        path=Path(tile['path']);array=np.load(path,mmap_mode='r',allow_pickle=False)
        if sha256(path)!=tile['sha256']:raise ValueError('Tile hash mismatch')
        if array.dtype.kind!='u' or list(array.shape)!=(np.array(tile['stop'])-tile['start']).tolist():raise ValueError('Invalid tile array')
        ids=array[array>0];lo=index*first['id_stride'];hi=(index+1)*first['id_stride']
        if len(ids) and (ids.min()<=lo or ids.max()>=hi):raise ValueError('Tile IDs outside reserved interval')
    image_coverage=None
    if a.image_manifest:
        manifest=json.loads(a.image_manifest.read_text())
        if manifest['files']['raw.npy']!=first['raw_sha256']:raise ValueError('Image manifest mismatches inference raw')
        validpath=a.image_manifest.parent/'image_valid_z.npy'
        if sha256(validpath)!=manifest['files']['image_valid_z.npy']:raise ValueError('Image validity hash mismatch')
        valid=np.load(validpath,mmap_mode='r');z0,z1=first['roi'][:2]
        image_coverage=dict(manifest_sha256=sha256(a.image_manifest),validity_sha256=sha256(validpath),
            observed_z_fraction=float(valid[z0:z1].mean()),unknown_output_z=np.flatnonzero(~valid[z0:z1]).tolist(),
            scope=manifest['image_valid']['scope'],all_defects_detected=False)
    out.mkdir(parents=True,exist_ok=False)
    seg=np.lib.format.open_memmap(out/'segmentation.npy',mode='w+',dtype=dtype,shape=shape);seg[:]=0
    start=np.asarray(first['roi'][::2]);coverage=[]
    for index,tile in expected.items():
        found=tiles.get(index);status='unstarted'
        if found:
            array=np.load(found['path'],mmap_mode='r');lo=np.array(tile['start'])-start;hi=np.array(tile['stop'])-start
            seg[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=array
            status='partial' if found['budget_exhausted'] else 'complete'
        coverage.append(dict(index=index,status=status,start=tile['start'],stop=tile['stop']))
    seg.flush();del seg
    complete=sum(t['status']=='complete' for t in coverage)
    write_json(out/'metadata.json',first['metadata'])
    write_json(out/'coverage.json',dict(tiles=coverage,complete_tiles=complete,total_tiles=len(expected),
        all_tiles_complete=complete==len(expected),image_observation=image_coverage))
    write_json(out/'receipt.json',dict(segmentation_sha256=sha256(out/'segmentation.npy'),
        inputs=[dict(path=str(p),sha256=sha256(p)) for p in a.receipts],metadata=first['metadata'],
        ground_truth_used=False,paper_comparable=False,all_tiles_complete=complete==len(expected),
        cross_tile_stitching=False,missing_tiles_encoded_zero=True,image_observation=image_coverage))

if __name__=='__main__':main()
