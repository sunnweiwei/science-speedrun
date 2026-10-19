"""Publish the selected data-only three-checkpoint package atomically."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256,write_json
from model_package import construct,verify


def export(checkpoints,output):
    import torch
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Export to a fresh staging name, then publish it')
    staging=Path(tempfile.mkdtemp(prefix='.model-',dir=output.parent))
    try:
        (staging/'weights').mkdir();models={};files={}
        for stage,path in checkpoints.items():
            cp=torch.load(path,map_location='cpu',weights_only=True)
            model=construct(stage,cp['architecture']);model.load_state_dict(cp['model'],strict=True)
            name=f'weights/{stage}.pt'
            torch.save({k:v for k,v in cp.items() if k!='optimizer'},staging/name)
            files[name]=sha256(staging/name)
            models[stage]=dict(architecture=cp['architecture'],source_sha256=sha256(path))
        write_json(staging/'package.json',dict(schema='pathfinder-model-package/v1',models=models,
            files=files,inference=dict(search_threshold=0.8)))
        verify(staging);os.rename(staging,output)
    except BaseException:
        shutil.rmtree(staging);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ffn','sense','shape','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();export({stage:getattr(a,stage) for stage in ('ffn','sense','shape')},a.output)
