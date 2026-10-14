import copy
import hashlib
import json
import unittest

from source_metadata import ARCO_URL, source_record, validate_source


class SourceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {'metadata': {
            '.zattrs': {'valid_time_start': '1940-01-01', 'valid_time_stop': '2026-04-30',
                        'valid_time_stop_era5t': '2026-08-29', 'last_updated': '2026-09-04'},
            'wind/.zarray': {'shape': [100, 721, 1440], 'chunks': [1, 721, 1440], 'dtype': '<f4'},
            'wind/.zattrs': {'units': 'm s**-1'},
        }}
        self.original = self.record(self.metadata)
        self.spec = {'url': ARCO_URL, 'metadata_sha256': self.original['metadata_sha256'],
                     'metadata_schema_sha256': self.original['metadata_schema_sha256'],
                     'required_valid_time_stop': '2022-12-31'}

    def record(self, metadata):
        return source_record(ARCO_URL, json.dumps(metadata).encode())

    def test_only_rolling_attributes_and_serialization_may_change(self):
        current = copy.deepcopy(self.metadata)
        current['metadata']['.zattrs'].update(last_updated='2026-09-14', valid_time_stop='2026-05-31',
                                             valid_time_stop_era5t='2026-09-08')
        validate_source(self.record(current), self.spec)
        validate_source(source_record(ARCO_URL, json.dumps(current, indent=2, sort_keys=True).encode()), self.spec)

    def test_rejects_schema_units_and_unknown_attribute_changes(self):
        for key, value in [('shape', [101, 721, 1440]), ('chunks', [2, 721, 1440]), ('dtype', '<f8')]:
            current = copy.deepcopy(self.metadata)
            current['metadata']['wind/.zarray'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'schema'):
                validate_source(self.record(current), self.spec)
        for key, value in [('units', 'knots'), ('unknown', 'new')]:
            current = copy.deepcopy(self.metadata)
            current['metadata']['wind/.zattrs'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'schema'):
                validate_source(self.record(current), self.spec)
        current = copy.deepcopy(self.metadata)
        current['metadata']['.zattrs']['valid_time_start'] = '2020-01-01'
        with self.assertRaisesRegex(ValueError, 'schema'):
            validate_source(self.record(current), self.spec)

    def test_rejects_insufficient_stable_coverage_and_missing_evidence(self):
        current = copy.deepcopy(self.metadata)
        current['metadata']['.zattrs']['valid_time_stop'] = '2021-12-31'
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_source(self.record(current), self.spec)
        record = self.original | {'metadata_sha256': 'changed'}
        del record['rolling_attributes']
        with self.assertRaisesRegex(ValueError, 'attributes'):
            validate_source(record, self.spec)

    def test_legacy_receipts_and_other_archives_remain_exactly_pinned(self):
        validate_source({'url': ARCO_URL, 'metadata_sha256': self.spec['metadata_sha256']}, self.spec)
        spec = {'url': 'gs://other/archive', 'metadata_sha256': hashlib.sha256(b'original').hexdigest()}
        validate_source(source_record(spec['url'], b'original'), spec)
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            validate_source(source_record(spec['url'], b'changed'), spec)
        with self.assertRaisesRegex(ValueError, 'URL changed'):
            validate_source(self.original | {'url': 'gs://other'}, self.spec)
