#!/usr/bin/env python3
"""Check frozen upstream metadata before spending time on builds or downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment" / "weathernext" / "wn2"))
from source_metadata import source_record, validate_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=30, help='Timeout per metadata request in seconds')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    lock = json.loads(Path(__file__).with_name('sources.lock.json').read_text())

    def check(item):
        name, spec = item
        url = spec['url'].replace('gs://', 'https://storage.googleapis.com/') + '/.zmetadata'
        result = {'source': name, 'expected_sha256': spec['metadata_sha256']}
        try:
            with urllib.request.urlopen(url, timeout=args.timeout) as response:
                record = source_record(spec['url'], response.read())
            result.update(actual_sha256=record['metadata_sha256'])
            try:
                validate_source(record, spec)
                result.update(status='passed', validation='exact_metadata' if record['metadata_sha256'] == spec['metadata_sha256'] else 'stable_schema_and_required_coverage')
            except ValueError as exc:
                result.update(status='changed', error=str(exc))
        except Exception as exc:
            result.update(status='unreachable', error=str(exc))
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check, lock['sources'].items()))
    passed = all(result['status'] == 'passed' for result in results)
    print(json.dumps({'status': 'passed' if passed else 'failed', 'sources': results,
                      'scope': 'Metadata availability and pins only; no data chunks, dependencies or GPU execution verified.'}, indent=2))
    if not passed:
        parser.exit(1, 'Source preflight failed. Do not change pins merely to bypass this check; use a validated matching snapshot or revalidate the dataset.\n')


if __name__ == '__main__':
    main()
