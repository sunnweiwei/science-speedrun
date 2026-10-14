#!/usr/bin/env python3
"""Summarize XLA allocations without double-counting values sharing buffers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def summarize(path):
    raw = path.read_bytes()
    text = raw.decode()
    total = re.search(r'Total bytes used: (\d+)', text)
    allocations = []
    for match in re.finditer(r'^allocation (\d+): size (\d+)([^\n]*)', text, re.M):
        number, size, flags = match.groups()
        category = ('parameter' if 'parameter ' in flags else
                    'constant' if 'constant' in flags else
                    'output' if 'maybe-live-out' in flags else 'temporary_or_other')
        allocations.append(dict(id=int(number), bytes=int(size), category=category,
                                flags=flags.strip(', :')))
    categories = {}
    for item in allocations:
        categories[item['category']] = categories.get(item['category'], 0) + item['bytes']
    if total is None or sum(categories.values()) != int(total[1]):
        raise ValueError(f'Allocation accounting does not match XLA total: {path}')
    peak = text.split(' (peak):', 1)
    peak_values = []
    if len(peak) == 2:
        for line in peak[1].splitlines():
            match = re.match(r'\s+(.+?): (\d+) bytes \(cumulative: (\d+) bytes\)', line)
            if match:
                name, size, cumulative = match.groups()
                peak_values.append(dict(name=name, bytes=int(size), cumulative_bytes=int(cumulative)))
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                total_allocation_bytes=int(total[1]), allocation_categories_bytes=categories,
                largest_allocations=sorted(allocations, key=lambda x:x['bytes'], reverse=True)[:20],
                peak_values_count=len(peak_values),
                largest_peak_values=sorted(peak_values, key=lambda x:x['bytes'], reverse=True)[:30],
                peak_values_note='Compiler live-range values may alias; do not add these to allocation totals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for phase in (args.run,args.run/'initial',args.run/'resumed',args.run/'cached')
                   for path in (phase/'xla-dump').glob('*buffer-assignment.txt'))
    if not paths:
        raise ValueError('No buffer assignment dumps: verify XLA module filter and compilation status')
    modules = [summarize(path) for path in paths]
    result = dict(scope='Per-executable compiler allocations, not a process-wide GPU memory ledger',
                  modules=sorted(modules, key=lambda x:x['total_allocation_bytes'], reverse=True))
    memory = args.run/'gpu-memory.csv'
    if memory.exists():
        with memory.open() as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        peaks = {}
        for row in rows:
            key = row['index']; value = int(row['memory.used [MiB]'].split()[0])
            if key not in peaks or value > peaks[key]['memory_mib']:
                peaks[key] = dict(memory_mib=value, timestamp=row['timestamp'])
        result['sampled_device_peaks'] = peaks
    result['limitations'] = ['Modules need not coexist: do not sum module totals.',
                            'Parameter category includes aliased outputs when present.',
                            'Device samples include other executables and runtime allocations.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
