"""Trusted CPU grading of a sealed prediction against a frozen local suite.

This does not execute submitted code or prove model/training compliance. Run in
an isolated verifier with suite/code/prediction mounted read-only.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json


def verify(prediction,suite,output):
    code=Path(__file__).resolve().parent;suite=suite.resolve();prediction=prediction.resolve()
    manifest=json.loads((suite/'suite.json').read_text())
    expected_files={'skeletons.json','metadata.json','run_length.py'}
    expected_code={'verify_prediction.py','evaluate_erl.py','prediction_contract.py','topology.py'}
    if (manifest.get('schema')!='pathfinder-evaluation-suite/v1' or manifest.get('primary_metric')!='nerl'
        or manifest.get('paper_comparable') is not False or manifest.get('boundary_margin_voxels')!=[0,0,0]
        or set(manifest['files'])!=expected_files or set(manifest['verifier_code'])!=expected_code):
        raise ValueError('Unsupported frozen evaluation suite')
    for name,digest in manifest['files'].items():
        path=suite/name
        if path.is_symlink() or not path.is_file() or sha256(path)!=digest:raise ValueError('Trusted suite integrity mismatch: '+name)
    for name,digest in manifest['verifier_code'].items():
        if sha256(code/name)!=digest:raise ValueError('Verifier version differs from frozen suite: '+name)
    if sha256(code.parent/'common.py')!=manifest['common_sha256']:raise ValueError('Verifier common code changed')
    if sha256(suite/'run_length.py')!='48f03d7c28b77853a431c0f163e9118befd057777a332be13cf500f72feb7f3c':
        raise ValueError('Unexpected upstream evaluator')
    assessment=output/'assessment';assessment.mkdir(exist_ok=False)
    result=subprocess.run([sys.executable,str(code/'evaluate_erl.py'),'--skeletons',str(suite/'skeletons.json'),
        '--metadata',str(suite/'metadata.json'),'--evaluator',str(suite/'run_length.py'),
        '--segmentation',str(prediction/'segmentation.npy'),'--prediction-contract',str(prediction),
        '--boundary-margin-voxels','0','0','0','--output',str(assessment/'metrics.json')],capture_output=True,text=True)
    (assessment/'evaluation.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('Independent evaluation failed; see assessment/evaluation.log')
    metrics=json.loads((assessment/'metrics.json').read_text());score=metrics['nerl']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1+1e-12:
        raise ValueError('Invalid normalized ERL')
    reward=min(1.,score)
    report=dict(status='scored',primary_metric='nerl',reward=reward,
        metrics_sha256=sha256(assessment/'metrics.json'),suite_sha256=sha256(suite/'suite.json'),
        prediction_manifest_sha256=sha256(prediction/'prediction.json'),paper_comparable=False,
        model_execution_verified=False,reference_scope=manifest['scope'],
        scope='Prediction scored against the frozen reference; training provenance and transfer protocol are not established by this verifier')
    write_json(output/'verification.json',report)
    (output/'reward.txt').write_text(f'{reward:.17g}\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prediction','suite','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();output=external(a.output);output.mkdir(parents=True,exist_ok=True)
    # Never leave an earlier nonzero reward behind after a failed check.
    (output/'reward.txt').write_text('0\n')
    try:verify(a.prediction,a.suite,output)
    except Exception as error:
        write_json(output/'verification.json',dict(status='failed',reward=0,error=str(error),paper_comparable=False))
        raise

if __name__=='__main__':main()
