"""Trusted operator: freeze local evaluation inputs separately from predictions."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def freeze(skeletons,metadata,evaluator,output):
    code=Path(__file__).resolve().parent;output=external(output)
    if sha256(evaluator)!='48f03d7c28b77853a431c0f163e9118befd057777a332be13cf500f72feb7f3c':
        raise ValueError('Pinned funlib evaluator required')
    output.mkdir(parents=True,exist_ok=False)
    for name,source in [('skeletons.json',skeletons),('metadata.json',metadata),('run_length.py',evaluator)]:
        shutil.copyfile(source,output/name)
    manifest=dict(schema='pathfinder-evaluation-suite/v1',primary_metric='nerl',
        files={n:sha256(output/n) for n in ('skeletons.json','metadata.json','run_length.py')},
        verifier_code={n:sha256(code/n) for n in ('verify_prediction.py','evaluate_erl.py','prediction_contract.py','topology.py')},
        common_sha256=sha256(code.parent/'common.py'),boundary_margin_voxels=[0,0,0],
        scope='Frozen local sparse skeletons; not official PATHFINDER protocol',paper_comparable=False)
    write_json(output/'suite.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('skeletons','metadata','evaluator','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();freeze(a.skeletons,a.metadata,a.evaluator,a.output)

if __name__=='__main__':main()
