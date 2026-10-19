import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from ffn_backend import select_backend, receipt_backend_matches


class BackendSelectionTests(unittest.TestCase):
    def test_norm32_installs_after_validation(self):
        install = Mock()
        model = object()
        with patch.dict(sys.modules, ffn_inference=SimpleNamespace(install_norm32=install)):
            self.assertEqual(select_backend(model, 'norm32'), 'norm32')
        install.assert_called_once_with(model)

    def test_no_silent_fallback_on_dependency_parity_or_execution_failure(self):
        for error in (ImportError('Triton unavailable'), ValueError('parity failed'), RuntimeError('CUDA failure')):
            with self.subTest(error=error), patch.dict(sys.modules, ffn_inference=SimpleNamespace(install_norm32=Mock(side_effect=error))):
                with self.assertRaises(type(error)):select_backend(object(), 'norm32')

    def test_eager_does_not_import_optional_dependencies(self):
        with patch.dict(sys.modules, ffn_inference=None):
            self.assertEqual(select_backend(object(), 'eager'), 'eager')

    def test_receipts_record_request_and_actual_backend(self):
        for backend in ('norm32', 'eager', 'norm32-graph'):
            row=dict(requested_inference_backend=backend, inference_backend=backend)
            self.assertTrue(receipt_backend_matches(row, backend))
        self.assertFalse(receipt_backend_matches(dict(requested_inference_backend='norm32', inference_backend='eager'), 'norm32'))
        self.assertTrue(receipt_backend_matches({}, 'eager'))

if __name__ == '__main__':unittest.main()
