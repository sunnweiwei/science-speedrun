#!/usr/bin/env python3
"""Prepare native MAGMa targets exclusively from the public development view."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def eligible(row):
    energy = row.get('collision_energy')
    return (row['adduct'] == '[M+H]+' and isinstance(energy, (int, float))
            and math.isfinite(energy) and row['instrument_type'] in {'Orbitrap', 'QTOF'})


def worker(item):
    import numpy as np
    from rdkit import Chem
    from ms_pred import common
    from ms_pred.magma.run_magma import magma_augmentation
    row, spectra_path = item
    name = row['query_id']
    mol = Chem.MolFromSmiles(row['target_smiles'])
    if mol is None or mol.GetNumHeavyAtoms() > 100:
        return name, None, 'invalid_or_over_100_heavy_atoms'
    if any(a.GetAtomicNum() not in common.VALID_ATOM_NUM for a in mol.GetAtoms()):
        return name, None, 'unsupported_atom'
    result = magma_augmentation(
        name, Path(spectra_path), {name: row['target_smiles']},
        {name: row['adduct']}, max_peaks=50, ppm_diff=20, merge_specs=False,
    )
    if result is None or not result[1]:
        return name, None, 'no_magma_fragment'
    if len(result[1]) != 1:
        raise ValueError(f'expected one condition for {name}')
    _, target = result[1][0]
    spectrum = common.MassSpec(
        row['collision_energy'], masses=np.asarray(row['mzs'], dtype=float),
        intens=np.asarray(row['intensities'], dtype=float),
    ).process_spec_file(parentmass=row['precursor_mz'])
    spectrum.denoise(max_num_inten=50, inten_thresh=0.001)
    tree = {
        'root_canonical_smiles': target.root_canonical_smiles,
        'collision_energy': target.collision_energy, 'adduct': row['adduct'],
        'frags': {str(i): {'frag': int(frag)} for i, frag in enumerate(target.int_frags)},
        'raw_spec': spectrum.spec.tolist(),
    }
    return name, json.dumps(tree, separators=(',', ':')), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=Path('/workspace/data'))
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('/opt/ms-pred'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-validation', type=int)
    args = p.parse_args()
    if args.workers < 1 or any(v is not None and v < 1 for v in [args.limit_train, args.limit_validation]):
        p.error('workers and supplied limits must be positive')
    sys.path.insert(0, str(args.source_root.resolve() / 'src'))
    from ms_pred import common
    started = time.monotonic()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = {name: digest(args.input_root / f'{name}.jsonl') for name in ['train', 'validation']}
    source_hash = hashlib.sha256()
    source_dir = args.source_root.resolve() / 'src'
    for path in sorted((source_dir / 'ms_pred').rglob('*.py')):
        source_hash.update(str(path.relative_to(source_dir)).encode())
        source_hash.update(path.read_bytes())
    config = {'source_files': sources, 'limits': [args.limit_train, args.limit_validation],
              'max_peaks': 50, 'ppm_diff': 20, 'domain': 'released_glacier_conditions',
              'native_source_sha256': source_hash.hexdigest()}
    config_path = root / 'preparation.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('preparation inputs/config changed; choose a new output directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            if digest(root / name) != expected:
                raise ValueError(f'prepared file hash mismatch: {name}')
        print(json.dumps(manifest, indent=2))
        return
    selected, counts, identities = [], {}, {}
    for fold, limit in [('train', args.limit_train), ('validation', args.limit_validation)]:
        rows = [json.loads(line) for line in (args.input_root / f'{fold}.jsonl').open() if line.strip()]
        identities[fold] = {r['target_inchikey'].split('-')[0] for r in rows}
        all_count = len(rows)
        rows = sorted((r for r in rows if eligible(r)),
                      key=lambda r: hashlib.sha256(r['query_id'].encode()).hexdigest())
        domain_count = len(rows)
        rows = rows[:limit] if limit is not None else rows
        counts[fold] = {'available': all_count, 'eligible': domain_count, 'selected': len(rows)}
        for row in rows:
            row = dict(row, preparation_fold=fold)
            selected.append(row)
    if identities['train'] & identities['validation']:
        raise ValueError('training and validation molecular identities overlap')
    names = [r['query_id'] for r in selected]
    if len(set(names)) != len(names) or not all(counts[f]['selected'] for f in counts):
        raise ValueError('empty fold or duplicate query identifiers')
    specs_path = root / 'spectra.hdf5'
    if not specs_path.exists():
        import numpy as np
        h5 = common.HDF5Dataset(specs_path.with_suffix('.partial'), mode='w')
        try:
            for row in selected:
                peaks = np.column_stack([row['mzs'], row['intensities']])
                value = common.spec_to_ms_str(
                    [(f"collision {row['collision_energy']}", peaks)],
                    {'parentmass': row['precursor_mz']},
                )
                h5.write_str(row['query_id'], value)
        finally:
            h5.close()
        os.replace(specs_path.with_suffix('.partial'), specs_path)
    database = root / 'targets.sqlite'
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE IF NOT EXISTS targets (query_id TEXT PRIMARY KEY, payload TEXT, reason TEXT)')
    done = {r[0] for r in db.execute('SELECT query_id FROM targets')}
    if done - set(names):
        raise ValueError('unexpected rows in resumable target database')
    pending = [(r, str(specs_path)) for r in selected if r['query_id'] not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, result in enumerate(executor.map(worker, pending, chunksize=8), 1):
            db.execute('INSERT INTO targets VALUES (?, ?, ?)', result)
            if i % 100 == 0:
                db.commit()
                print(json.dumps({'prepared': i + len(done), 'total': len(names),
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
    db.commit()
    if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('target database integrity failure')
    reasons = dict(db.execute('SELECT reason, COUNT(*) FROM targets WHERE payload IS NULL GROUP BY reason'))
    annotated = {r[0] for r in db.execute('SELECT query_id FROM targets WHERE payload IS NOT NULL')}
    for fold in counts:
        rows = [r for r in selected if r['preparation_fold'] == fold and r['query_id'] in annotated]
        counts[fold]['annotated'] = len(rows)
        if not rows:
            raise ValueError(f'no annotated rows in {fold}')
        with (root / f'{fold}.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
    db.close()
    files = {name: digest(root / name) for name in ['targets.sqlite', 'train.jsonl', 'validation.jsonl']}
    manifest = {'schema': 'massspecgym-glacier-training/v1', 'config': config, 'counts': counts,
                'excluded_reasons': reasons, 'files': files,
                'elapsed_seconds': time.monotonic() - started,
                'policy': 'Train for fitting; validation for selection only; no test inputs consumed.'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    root.chmod(root.stat().st_mode | 0o555)
    for path in root.iterdir():
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o444)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
