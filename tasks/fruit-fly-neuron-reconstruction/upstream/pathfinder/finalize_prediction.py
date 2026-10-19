"""Create a sealed reconstruction from model inference, without evaluation GT.

This does not execute the models again or independently prove their provenance.
It checks recorded input integrity and preserves coverage for separate review.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json
from prediction_contract import seal
from context_storage import segmentation_path


def finalize(root, inference, output, threshold=.2, mode='motif'):
    if type(threshold) not in (int,float) or not 0<=threshold<=1:raise ValueError('Invalid search threshold')
    if mode not in ('motif','disjoint-pairs'):raise ValueError('Invalid reconciliation mode')
    root = external(root); output = external(output)
    if root not in output.parents:
        raise ValueError('Output must stay under original budget root')
    code = Path(__file__).resolve().parent
    inference = Path(inference).resolve()
    read = lambda p: json.loads(p.read_text())
    receipt_path = inference/'receipt.json'
    receipt = read(receipt_path)
    plan_path = inference/'plan.json'
    scores_path = inference/'shape/merged.json'
    for path, key in ((plan_path, 'plan_sha256'), (scores_path, 'scores_sha256')):
        if sha256(path) != receipt[key]:
            raise ValueError('Inference integrity mismatch: '+key)
    plan = read(plan_path)
    assembled = Path(plan['input_paths']['assembled'])
    if not assembled.is_absolute():
        raise ValueError('Explicit absolute assembly path required')
    if sha256(assembled/'receipt.json') != plan['assembly_receipt_sha256']:
        raise ValueError('Assembly changed after inference')
    scores = read(scores_path)
    if scores.get('ground_truth_used') is not False or scores.get('allow_partial_context') is not False:
        raise ValueError('Prediction-only strict-context inference required')
    # Read required coverage before creating any outputs. Identity and incomplete
    # coverage remain valid artifacts, never a claim of completed reconstruction.
    coverage = read(assembled/'coverage.json')
    candidates = read(inference/'sense/edges.json')
    subprocess.run([sys.executable, str(code/'reconcile.py'), '--scores', str(scores_path),
        '--segmentation', str(segmentation_path(assembled)),
        '--instance-index', str(assembled/'instance-index.json'),
        '--root', str(root), '--output', str(output), '--threshold', str(threshold),
        '--mode', mode], check=True)
    check = seal(output/'segmentation.npy', assembled/'metadata.json')
    search = read(output/'receipt.json')
    result = dict(schema='pathfinder-finalized-prediction/v1',
        search_threshold=threshold,reconciliation_mode=mode,prediction_manifest_sha256=check['manifest_sha256'],
        search_receipt_sha256=sha256(output/'receipt.json'),
        inference_receipt_sha256=sha256(receipt_path),
        inputs={str(p): sha256(p) for p in (plan_path, scores_path,
            assembled/'metadata.json', assembled/'coverage.json', inference/'sense/edges.json')},
        learned_merges_present=search['remapped_ids'] > 0,
        remapped_ids=search['remapped_ids'], retained_edges=len(search['retained']),
        prediction_coverage=coverage, candidate_coverage=candidates['coverage'],
        scored_full_context_motifs=scores['full_context_scored'],
        unprocessed_motifs=len(scores['unprocessed_motifs']),
        motif_enumeration_truncated=scores['truncated'],
        ground_truth_read=False, model_execution_independently_verified=False,
        paper_comparable=False,
        implementation_sha256={name: sha256(code/name) for name in
            ('finalize_prediction.py', 'reconcile.py', 'search.py', 'lazy_search.py', 'prediction_contract.py')},
        scope='Sealed prediction only; independent quality and coverage review required')
    write_json(output/'finalization.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'inference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument("--threshold",type=float,default=.2)
    p.add_argument('--mode',choices=['motif','disjoint-pairs'],default='motif')
    args = p.parse_args()
    finalize(args.root, args.inference, args.output, args.threshold,args.mode)


if __name__ == '__main__':
    main()
