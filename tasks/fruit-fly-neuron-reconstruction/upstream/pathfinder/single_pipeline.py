"""Fresh, prediction-only single-checkpoint pipeline; no suite or scheduler input.

Uses the same low-level algorithms as the recorded native fast ablation.
The enclosing verifier provides trusted code/FPS and a hard whole-run timeout.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import signal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sha256, write_json
from execution_resources import allocation

CODE = Path(__file__).resolve().parent
# FFN workers per visible GPU, run under CUDA MPS: one worker leaves the GPU mostly idle (its
# flood-filling loop is single-threaded CPU work around batches of four 33^3 views), and without
# MPS several processes only time-share the GPU. Window outputs do not depend on the worker count.
FFN_WORKERS_PER_GPU = 4
# SENSE, FFN-pair baseline and SHAPE shards per visible GPU (also under MPS). Every stage assigns
# items round-robin by ordinal and merges in rank order, so results do not depend on the shard count.
DOWNSTREAM_WORKERS_PER_GPU = 4


def start_mps(logs):
    """Start a private CUDA MPS daemon for this run; return a stop function (no-op without MPS)."""
    import shutil
    control = shutil.which('nvidia-cuda-mps-control')
    if control is None:
        return lambda: None
    pipe = Path('/tmp/pathfinder-mps'); pipe.mkdir(exist_ok=True); (pipe / 'log').mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(pipe / 'log'))
    with (logs / 'mps.log').open('a') as log:
        if subprocess.run([control, '-d'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode:
            return lambda: None
    os.environ.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(pipe / 'log'))
    def stop():
        subprocess.run([control], input=b'quit\n', env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return stop


def command(script, **options):
    result = [sys.executable, '-B', str(CODE / script)]
    for key, value in options.items():
        if value is False or value is None:
            continue
        result.append('--' + key.replace('_', '-'))
        if value is not True:
            result.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return result


def run(root, runtime, limits, ffn_inference_backend="norm32"):
    if ffn_inference_backend not in ("eager", "norm32", "norm32-graph"):
        raise ValueError("Unknown FFN inference backend")
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    image = root / 'image/manifest.json'; package = root / 'model'
    from model_package import verify
    check = verify(package)
    if 'ffn_ensemble' in check or check['search_threshold'] != 0.8:
        raise ValueError('One FFN checkpoint and fixed SHAPE threshold 0.8 required')
    required = {'ffn_seconds', 'sense_seconds', 'baseline_seconds', 'shape_seconds',
                'memory_gib', 'max_instances', 'numeric_max_label', 'window_bytes'}
    if set(limits) != required or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError('Explicit positive integer single-protocol limits required')
    masks = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    if not masks:
        masks = ['0', '1', '2', '3']
    allocation(root, sum(limits[k] for k in required if k.endswith('_seconds')) + 600, gpus=len(masks))
    out = root / 'prediction'; out.mkdir()
    logs = out / 'logs'; logs.mkdir()
    source_pins = {str(p): sha256(p) for p in [CODE.parent/'common.py', *CODE.glob('*.py')]}
    records = []
    record_lock = threading.Lock()
    process_lock = threading.Lock()
    processes = set()
    failed = threading.Event()

    def stop_all():
        failed.set()
        with process_lock:
            for process in list(processes):
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass

    def execute(name, cmd, mask=None):
        start = time.monotonic()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        if mask is not None: env['CUDA_VISIBLE_DEVICES'] = mask
        with (logs / (name + '.log')).open('x') as log:
            with process_lock:
                if failed.is_set(): raise RuntimeError('Earlier pipeline stage failed')
                process=subprocess.Popen(['/usr/bin/time', '-v', '-o', str(logs/(name+'.resources.txt')), *cmd],
                    env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,start_new_session=True)
                processes.add(process)
            try:
                result=process.wait()
                if result:
                    stop_all()
                    raise subprocess.CalledProcessError(result,cmd)
            finally:
                with process_lock: processes.discard(process)
        with record_lock:
            records.append(dict(stage=name, command=cmd, wall_seconds=time.monotonic()-start))
            write_json(out/'execution.json', dict(complete=False, stages=records))

    def parallel(stage, commands):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
            futures = [pool.submit(execute, f'{stage}-{i}', cmd, masks[i % len(masks)]) for i, cmd in enumerate(commands)]
            for f in concurrent.futures.as_completed(futures): f.result()

    meta = json.loads(image.read_text())
    roi = [v for n in meta['shape'] for v in (16, n-16)]
    # Checkpoint-independent seeds for every window, computed in parallel on all CPUs.
    execute('seeds', command('precompute_seeds.py', root=root, image_manifest=image, cache=out/'seed-cache',
        receipt=out/'seed-cache.json', roi=roi, core_size=436, margin=32, clip_boundary_to_image=True))
    stop_mps = start_mps(logs)
    execute('ffn', command('launch_overlap.py', root=root, output=out/'inference',
        workers_per_gpu=FFN_WORKERS_PER_GPU, seed_cache=out/'seed-cache', assignment_policy='voxel-balanced-v1',
        inference_backend=ffn_inference_backend,
        image_manifest=image, checkpoint=package/'weights/ffn.pt', roi=roi, core_size=436,
        margin=32, batch=4, seconds=limits['ffn_seconds'], max_calls=1000000,
        seed_order='ascending', clip_boundary_to_image=True,
        compressed_window_bytes=limits['window_bytes'], compression_level=6))
    execute('stitch', command('stitch_ffn.py', root=root, manifest=out/'inference/manifest.json',
        output=out/'stitched', memory_gib=limits['memory_gib'],
        compressed_core_bytes=limits['window_bytes'], compression_level=6, mapping_bytes=1000000000,
        ffn_source=runtime/'ffn', connectomics_source=runtime/'connectomics'))
    execute('context', command('prepare_prediction_context.py', root=root, prediction=out/'stitched',
        manifest=out/'inference/manifest.json', image_manifest=image, output=out/'context',
        memory_gib=limits['memory_gib'], max_instances=limits['max_instances'],
        numeric_max_label=limits['numeric_max_label'], reference_parent_array=True))
    context = out/'context'; base = out/'stitched/segmentation.npy'; n = len(masks) * DOWNSTREAM_WORKERS_PER_GPU
    from prepare_prediction_context import audit
    from index_storage import load_index, select_table_centers
    coverage = audit(context)
    idx = load_index(context/'instance-index.json')
    selected, *_ = select_table_centers(idx['records'], idx['shape'], 0, 1)
    eligible = len(selected); idx['records'].table._mmap.close()
    expected = dict(segmentation_sha256=coverage['segmentation_sha256'],
        image_manifest_sha256=sha256(image), instance_index_sha256=sha256(context/'instance-index.json'),
        assembled_receipt_sha256=sha256(context/'receipt.json'), coverage_sha256=sha256(context/'coverage.json'),
        checkpoint_sha256=sha256(package/'weights/sense.pt'), all_tiles_complete=True)
    write_json(out/'expected-sense.json', expected)
    parallel('sense', [command('infer_continuous_sense.py', image_manifest=image, assembled=context,
        index=context/'instance-index.json', checkpoint=package/'weights/sense.pt',
        output=out/f'sense/shard-{i}', shards=n, shard_index=i, batch=8,
        seconds=limits['sense_seconds'], max_calls=max(1, len(range(i, eligible, n)))) for i in range(n)])
    edges = out/'refinement/sense/edges.json'
    execute('merge-sense', command('merge_inference_shards.py', stage='sense',
        shards=[out/f'sense/shard-{i}' for i in range(n)], output=edges, stream_sense=True,
        require_complete=True, expected_eligible=eligible, expected_provenance=out/'expected-sense.json'))
    execute('proposals', command('shared_proposals.py', edges=edges, base=base, output=out/'decisions.json'))
    count = len(json.loads((out/'decisions.json').read_text())['decisions'])
    # Manifest origin is the raw-grid origin, not the cropped prediction origin.
    write_json(out/'image-metadata.json', dict(meta, origin_nm=meta['origin_nm']))
    parallel('baseline', [command('run_ffn_agglomeration.py', raw=image.parent/'raw.npy', base=base,
        image_metadata=out/'image-metadata.json', base_metadata=context/'metadata.json',
        decisions=out/'decisions.json', checkpoint=package/'weights/ffn.pt',
        output=out/f'baseline/shard-{i}.json', batch=8, seconds=limits['baseline_seconds'],
        shards=n, shard_index=i) for i in range(n)])
    execute('merge-baseline', command('merge_ffn_agglomeration.py', decisions=out/'decisions.json',
        shards=[out/f'baseline/shard-{i}.json' for i in range(n)], output=out/'baseline.json',
        model_package=package, rule='pathfinder2025'))
    geometry = json.loads((context/'metadata.json').read_text())
    parallel('shape', [command('infer_shape.py', root=root, checkpoint=package/'weights/shape.pt',
        segments=base, edges=edges, output=out/f'shape/shard-{i}.json',
        voxel_size_nm=geometry['voxel_size_nm'], origin_nm=geometry['origin_nm'], max_nodes=2,
        max_motifs=max(1,count), instance_index=context/'instance-index.json',
        coverage=context/'coverage.json', metadata=context/'metadata.json', seconds=limits['shape_seconds'],
        shards=n, shard_index=i) for i in range(n)])
    stop_mps()
    for i in range(n):
        row=json.loads((out/f'shape/shard-{i}.json').read_text())
        if row['budget_exhausted'] or row['truncated'] or len(row['motifs'])!=len(range(i,count,n)):
            raise RuntimeError('Incomplete SHAPE shard')
    merged = out/'refinement/shape/merged.json'
    execute('merge-shape', command('merge_inference_shards.py', stage='shape',
        shards=[out/f'shape/shard-{i}.json' for i in range(n)], edges=edges, output=merged))
    scores=json.loads(merged.read_text())
    if scores['truncated'] or scores['unprocessed_motifs']:
        raise RuntimeError('Incomplete SHAPE enumeration')
    refinement=out/'refinement'
    write_json(refinement/'plan.json', dict(input_paths=dict(assembled=str(context)),
        assembly_receipt_sha256=sha256(context/'receipt.json')))
    write_json(refinement/'receipt.json', dict(plan_sha256=sha256(refinement/'plan.json'), scores_sha256=sha256(merged)))
    execute('finalize', command('finalize_prediction.py', root=root, inference=refinement,
        output=out/'submission', threshold=0.8, mode='disjoint-pairs'))
    for path, digest in source_pins.items():
        if sha256(path)!=digest: raise RuntimeError('Trusted source changed during inference')
    write_json(out/'execution.json', dict(complete=True, protocol='single-checkpoint-fast/v1', ffn_inference_backend=ffn_inference_backend,
        stages=records, source_sha256=source_pins, package_manifest_sha256=check['package_manifest_sha256'],
        eligible_sense_centers=eligible, shared_candidates=count, checkpoint_count=1,
        seed_order='ascending', consensus=False, ground_truth_used=False,
        prediction_sha256=sha256(out/'submission/segmentation.npy')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','runtime','limits'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument("--ffn-inference-backend", choices=["eager", "norm32", "norm32-graph"], default="norm32")
    args=parser.parse_args();run(args.root,args.runtime,json.loads(args.limits.read_text()),args.ffn_inference_backend)
