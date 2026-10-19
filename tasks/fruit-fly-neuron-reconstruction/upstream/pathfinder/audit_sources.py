"""Verify pinned public components; do not label them a PATHFINDER release."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json

PINS = {
    'google-ffn': dict(commit='b8df2d96d7c8da057fd2702ced19732a08ab6ae4', url='https://github.com/google/ffn',
        files=['train.py','run_inference.py','README.md','LICENSE'], role='Public FFN component; PATHFINDER FFN v1.5 parity unverified'),
    'google-connectomics': dict(commit='bf2f6caf083bb5d459ec86f5e859c366857be1ea', url='https://github.com/google-research/connectomics',
        files=['connectomics/mogen/flow_matching/train.py','connectomics/mogen/flow_matching/inference.py',
               'connectomics/mogen/models/pointinfinity.py','LICENSE'], role='MoGen public source; not the SENSE/SHAPE reconstruction pipeline'),
}
MISSING = ['verified PATHFINDER release and runnable configuration',
    'FFN v1.5 / SENSE / SHAPE matching trained checkpoints',
    'IBEAM-mSEM EM and proofread evaluation reconstruction',
    'author train/validation/test spatial definitions and preprocessing',
    'author evaluation implementation or numerical parity fixtures']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--require-replay-ready', action='store_true')
    a = p.parse_args(); out = external(a.output)
    verified = {}
    for name, pin in PINS.items():
        repo = a.root/'vendor'/name
        def git(*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
        if git('rev-parse','HEAD') != pin['commit'] or git('status','--porcelain','--untracked-files=no'):
            raise ValueError('Unpinned or modified source: '+name)
        checked = {}
        for relative in pin['files']:
            path = repo/relative
            if path.suffix == '.py': ast.parse(path.read_text(),filename=str(path))
            checked[relative] = sha256(path)
        names = git('ls-tree','-r','--name-only','HEAD').splitlines()
        verified[name] = dict(**pin, files_sha256=checked, tracked_files=len(names),
            pathfinder_named_paths=[x for x in names if 'pathfinder' in x.lower()])
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_audit_passed=True, components=verified, official_pathfinder_replay_ready=False,
        missing_release_artifacts=MISSING, author_model_replayed=False, sota_gap_measured=False,
        availability='No complete public PATHFINDER release verified in this audit; absence is not proof of permanent closure',
        paper_release_statement_url='https://www.researchgate.net/publication/391957338_Accelerating_Neuron_Reconstruction_with_PATHFINDER',
        paper_doi='10.1101/2025.05.16.654254',
        paper_table1_nerl=.942, paper_metric_is_not_cremi_partner_f1=True)
    write_json(out, result)
    print(json.dumps(result, indent=2))
    if a.require_replay_ready: raise SystemExit(2)

if __name__ == '__main__': main()
