"""Partition public input by the native model's actual inference support."""

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem


def supports_pool(candidates, valid_atoms):
    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate['smiles'])
        if mol is None or any(a.GetAtomicNum() not in valid_atoms for a in mol.GetAtoms()):
            continue
        try:
            if Chem.RemoveHs(mol).GetNumAtoms() <= 100:
                return True
        except Exception:
            continue
    return False


def supports_observation(query):
    # Match the native scorer's precursor filter and 1,500 Da binning range.
    return any(intensity > 0 and 0 <= mz < 1500 and mz <= query['precursor_mz'] + 1
               for mz, intensity in zip(query['mzs'], query['intensities']))


def partition(input_root, output_root, valid_atoms, released_domain):
    with (input_root / 'queries.jsonl').open() as stream:
        queries = [json.loads(line) for line in stream if line.strip()]
    with (input_root / 'candidate_pools.jsonl').open() as stream:
        pools = {row['candidate_pool_id']: row for row in
                 (json.loads(line) for line in stream if line.strip())}
    support = {}
    groups = {'glacier': [], 'fallback': []}
    for query in queries:
        eligible = released_domain(query) and supports_observation(query)
        if eligible:
            pool_id = query['candidate_pool_id']
            if pool_id not in support:
                support[pool_id] = supports_pool(pools[pool_id]['candidates'], valid_atoms)
            eligible = support[pool_id]
        groups['glacier' if eligible else 'fallback'].append(query)
    output_root.mkdir(parents=True, exist_ok=False)
    for name, rows in groups.items():
        target = output_root / name
        target.mkdir()
        used = {q['candidate_pool_id'] for q in rows}
        for filename, values in [('queries.jsonl', rows),
                                 ('candidate_pools.jsonl', [pools[k] for k in sorted(used)])]:
            with (target / filename).open('w') as stream:
                for row in values:
                    stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        manifest = {'schema': 'massspecgym-retrieval-input/v1',
                    'source': 'native model support partition; public input only',
                    'counts': {'queries': len(rows), 'candidate_pools': len(used)},
                    'files': {n: hashlib.sha256((target / n).read_bytes()).hexdigest()
                              for n in ['queries.jsonl', 'candidate_pools.jsonl']}}
        (target / 'manifest.json').write_text(json.dumps(manifest) + '\n')
    (output_root / 'glacier-query-ids.txt').write_text(''.join(q['query_id'] + '\n' for q in groups['glacier']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    from glacier_contract import is_released_domain
    from ms_pred.common.chem_utils import VALID_ATOM_NUM
    partition(args.input_root, args.output_root, set(VALID_ATOM_NUM), is_released_domain)
