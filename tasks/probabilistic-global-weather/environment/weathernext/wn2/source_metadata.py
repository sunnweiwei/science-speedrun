"""Preserve archive schema identity while allowing declared ARCO rolling attributes."""
import copy
from datetime import date
import hashlib
import json

ARCO_URL = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
ROLLING_ATTRIBUTES = ('last_updated', 'valid_time_stop', 'valid_time_stop_era5t')


def source_record(url, raw):
    record = {'url': url, 'metadata_sha256': hashlib.sha256(raw).hexdigest()}
    if url == ARCO_URL:
        metadata = json.loads(raw)
        canonical = copy.deepcopy(metadata)
        attrs = canonical['metadata']['.zattrs']
        record['rolling_attributes'] = {key: attrs.pop(key) for key in ROLLING_ATTRIBUTES}
        record['metadata_schema_sha256'] = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return record


def validate_source(record, spec):
    if record.get('url') != spec['url']:
        raise ValueError('Weather archive URL changed')
    # Preserve acceptance of already materialized receipts from the original snapshot.
    if record.get('metadata_sha256') == spec['metadata_sha256']:
        return
    if spec['url'] != ARCO_URL or 'metadata_schema_sha256' not in spec:
        raise ValueError('Weather archive metadata changed')
    if record.get('metadata_schema_sha256') != spec['metadata_schema_sha256']:
        raise ValueError('ARCO archive schema changed beyond the allowed rolling attributes')
    attrs = record.get('rolling_attributes', {})
    if set(attrs) != set(ROLLING_ATTRIBUTES) or not all(isinstance(v, str) and v for v in attrs.values()):
        raise ValueError('Missing or invalid ARCO rolling attributes')
    if date.fromisoformat(attrs['valid_time_stop']) < date.fromisoformat(spec['required_valid_time_stop']):
        raise ValueError('ARCO stable ERA5 coverage no longer includes the required dates')
    if date.fromisoformat(attrs['valid_time_stop_era5t']) < date.fromisoformat(attrs['valid_time_stop']):
        raise ValueError('ARCO ERA5T coverage precedes stable ERA5 coverage')
