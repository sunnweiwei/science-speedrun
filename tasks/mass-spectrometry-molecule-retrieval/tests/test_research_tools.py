"""Regression tests for interrupted publication and scientific feedback."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'environment/evaluation'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publication = load('publication', ROOT / 'environment/baseline/starter/publication.py')
diagnostics = load('diagnostics', ROOT / 'environment/evaluation/diagnostics.py')
sys.modules['diagnostics'] = diagnostics
score = load('validation_score', ROOT / 'environment/evaluation/score.py')
paired = load('paired_score', ROOT / 'environment/evaluation/paired_bootstrap.py')
sys.path.insert(0, str(ROOT / 'tests'))
import main_verifier


def build(stage, marker):
    (stage / 'README.md').write_text(marker)
    (stage / 'weights.pt').write_bytes(marker.encode())
    (stage / 'config.json').write_text('{}')
    (stage / 'infer.sh').write_text('#!/bin/bash\nprintf "%s\\n" ' + marker + '\n')
    (stage / 'model_manifest.json').write_text(json.dumps({
        'schema': 'massspecgym-retrieval-model/v1',
        'checkpoint': {'path': 'weights.pt', 'sha256': hashlib.sha256(marker.encode()).hexdigest(), 'format': 'pytorch'},
        'model_config': 'config.json', 'training_provenance': {'data': 'fixture', 'pretrained_assets': []},
    }))


class NativePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = load('native_publish', ROOT / 'environment/baseline/starter/glacier/publish.py')
        self.checkpoint = self.root / 'trained.ckpt'
        self.state = {
            'state_dict': {'fixture.weight': torch.tensor([1.0, 2.0])},
            'hyper_parameters': json.loads((ROOT / 'environment/baseline/starter/glacier/config.json').read_text()),
            'pytorch-lightning_version': '2.6.5', 'global_step': 10, 'epoch': 1,
            'task_training': {'initialization': 'random', 'batch_size': 8,
                              'data_manifest_sha256': 'a' * 64},
        }
        torch.save(self.state, self.checkpoint)
        self.fallback = self.root / 'fallback'
        self.fallback.mkdir()
        build(self.fallback, 'trained-fallback-fixture')
        self.source = self.root / 'source'
        (self.source / 'src/ms_pred').mkdir(parents=True)
        (self.source / 'src/ms_pred/__init__.py').write_text('# source fixture\n')
        (self.source / 'LICENSE').write_text('fixture license\n')

    def command(self, destination: Path) -> list[str]:
        return [sys.executable, str(ROOT / 'environment/baseline/starter/glacier/publish.py'),
                '--checkpoint', str(self.checkpoint), '--fallback', str(self.fallback),
                '--source-root', str(self.source), '--output-dir', str(destination)]

    def assert_native_invocation(self, generation: Path, batch_size: int) -> None:
        """Exercise the published adapter with bounded stand-ins for model execution."""
        native_infer = load('published_native_infer', generation / 'infer.py')
        input_root, output_root = self.root / 'public-input', self.root / 'rankings'

        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            script = Path(command[1]).name
            if script == 'stage_input.py':
                partition = Path(command[command.index('--output-root') + 1])
                for branch in ('glacier', 'fallback'):
                    (partition / branch).mkdir(parents=True)
                    (partition / branch / 'queries.jsonl').write_text('{}\n')
            elif script == 'prepare_glacier_candidates.py':
                Path(command[command.index('--output') + 1]).write_text('smiles\nCCO\n')
            elif script == 'merge_glacier_fallback.py':
                (output_root / 'runtime.json').write_text('{}\n')
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(sys, 'argv', ['infer.py', str(input_root), str(output_root)]), \
                mock.patch.object(native_infer.subprocess, 'run', side_effect=run) as execute:
            native_infer.main()
        commands = [call.args[0] for call in execute.call_args_list]
        predictor = next(command for command in commands if Path(command[1]).name == 'predict_smis_joint.py')
        for option, value in {'--batch-size': batch_size, '--num-cpu-workers': 2,
                              '--feat-cache-size': 64, '--prefetch-factor': 1}.items():
            self.assertEqual(predictor[predictor.index(option) + 1], str(value))
        self.assertEqual(predictor[predictor.index('--checkpoint') + 1], str(generation / 'glacier.ckpt'))
        scorer = next(command for command in commands if Path(command[1]).name == 'score_glacier_predictions.py')
        self.assertEqual(scorer[scorer.index('--distance') + 1], 'entropy')
        self.assertTrue(any(command[:2] == ['bash', str(generation / 'fallback/infer.sh')] for command in commands))
        self.assertFalse((output_root / 'runtime.json').exists())

    def test_default_and_cli_override_preserve_model_and_training(self) -> None:
        import torch
        configs = []
        for name, override, batch_size in [('api-default', None, 64),
                                           ('cli-default', None, 64), ('cli-small', 8, 8)]:
            with self.subTest(name=name):
                destination = self.root / name
                if name == 'api-default':
                    self.native.publish(self.checkpoint, self.fallback, self.source, destination)
                else:
                    command = self.command(destination)
                    if override is not None:
                        command.extend(['--inference-batch-size', str(override)])
                    subprocess.run(command, check=True, capture_output=True, text=True)
                # Publication must remain portable, including its selected generation.
                relocated = self.root / (name + '-relocated')
                shutil.move(destination, relocated)
                manifest = main_verifier.validate_submission(relocated)
                config = json.loads((relocated / manifest['model_config']).read_text())
                configs.append(config)
                self.assertEqual(config['batch_size'], batch_size)
                self.assertEqual(config['training'], self.state['task_training'])
                self.assertEqual(config['method'], 'native-glacier-with-trained-deepsets-fallback')
                self.assertEqual(config['routing'], 'released_conditions_with_usable_observation_and_supported_pool')
                self.assertEqual(manifest['training_provenance'], {
                    'data': 'MassSpecGym v1.5 train; validation for selection only', 'pretrained_assets': []})
                saved = torch.load(relocated / manifest['checkpoint']['path'], weights_only=False)
                self.assertTrue(torch.equal(saved.pop('state_dict')['fixture.weight'],
                                            self.state['state_dict']['fixture.weight']))
                self.assertEqual(saved, {key: value for key, value in self.state.items() if key != 'state_dict'})
                generation = (relocated / manifest['model_config']).parent
                for name in ['stage_input.py', 'config.json', 'train.py']:
                    source = ROOT / 'environment/baseline/starter/glacier' / name
                    target = generation / ('stage_input.py' if name == 'stage_input.py' else 'training/' + name)
                    self.assertEqual(target.read_bytes(), source.read_bytes())
                self.assertEqual((generation / 'fallback/weights.pt').read_bytes(),
                                 (self.fallback / 'weights.pt').read_bytes())
                self.assert_native_invocation(generation, batch_size)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], {**configs[2], 'batch_size': 64})

    def test_invalid_api_batch_preserves_existing_publication(self) -> None:
        destination = self.root / 'published'
        self.native.publish(self.checkpoint, self.fallback, self.source, destination)
        before = (destination / 'model_manifest.json').read_bytes()
        for value in (0, -1, 1.5, '8', True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                self.native.publish(self.checkpoint, self.fallback, self.source, destination,
                                    inference_batch_size=value)
            self.assertEqual((destination / 'model_manifest.json').read_bytes(), before)

    def test_invalid_cli_batch_fails_before_publication(self) -> None:
        destination = self.root / 'invalid'
        for value in ('0', '-1', '1.5', 'invalid'):
            with self.subTest(value=value):
                result = subprocess.run(self.command(destination) + ['--inference-batch-size', value],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('inference batch size must be a positive integer', result.stderr)
                self.assertFalse(destination.exists())


class ResearchToolsTests(unittest.TestCase):
    def test_native_routes_unsupported_valid_inputs_to_the_trained_fallback(self):
        native = load('native_stage', ROOT / 'environment/baseline/starter/glacier/stage_input.py')
        self.assertTrue(native.supports_pool([{'smiles': 'CCO'}], {1, 6, 7, 8}))
        self.assertFalse(native.supports_pool([{'smiles': 'C' * 101}], {1, 6, 7, 8}))
        self.assertFalse(native.supports_observation(dict(mzs=[1500.0], intensities=[1.0], precursor_mz=200.0)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input'; source.mkdir()
            ordinary = dict(query_id='ordinary', candidate_pool_id='p', mzs=[50.0],
                            intensities=[1.0], precursor_mz=100.0)
            queries = [ordinary, dict(ordinary, query_id='outside', mzs=[1500.0]),
                       dict(ordinary, query_id='large-pool', candidate_pool_id='large')]
            pools = [{'candidate_pool_id': 'p', 'candidates': [{'candidate_id': 'c', 'smiles': 'CCO'}]},
                     {'candidate_pool_id': 'large', 'candidates': [{'candidate_id': 'large-c', 'smiles': 'C' * 101}]}]
            for filename, rows in [('queries.jsonl', queries), ('candidate_pools.jsonl', pools)]:
                (source / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            native.partition(source, root / 'output', {1, 6, 7, 8}, lambda query: True)
            self.assertEqual((root / 'output/glacier-query-ids.txt').read_text(), 'ordinary\n')
            fallback = [json.loads(line) for line in (root / 'output/fallback/queries.jsonl').read_text().splitlines()]
            self.assertEqual({row['query_id'] for row in fallback}, {'outside', 'large-pool'})
            self.assertEqual(len(fallback) + 1, len(queries))

    def test_resuming_fit_state_does_not_refund_completed_validation(self):
        native = load('native_train', ROOT / 'environment/baseline/starter/glacier/train.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # A fit checkpoint precedes expensive final validation/export.
            self.assertEqual(native.charged_elapsed(root, 190), 190)
            (root / 'result.json').write_text(json.dumps({'elapsed_seconds': 567}))
            self.assertEqual(native.charged_elapsed(root, 190), 567)
            # A later interrupted segment must retain its greater usage too.
            self.assertEqual(native.charged_elapsed(root, 800), 800)
            (root / 'budget-ledger.json').write_text(json.dumps({'elapsed_seconds': 900}))
            self.assertEqual(native.charged_elapsed(root, 800), 900)

    def test_unparseable_candidate_has_a_finite_fallback_but_training_stays_strict(self):
        import torch
        starter = load('starter_common', ROOT / 'environment/baseline/starter/common.py')
        # Invalid in every RDKit version exercises the same boundary as legacy
        # valences accepted by the task's RDKit 2023 but rejected by 2025.
        with self.assertRaisesRegex(ValueError, 'invalid candidate SMILES'):
            starter.morgan_fingerprint('not_a_smiles')
        fp = starter.morgan_fingerprint('not_a_smiles', allow_unparseable=True)
        self.assertTrue(torch.isfinite(fp).all())
        self.assertEqual(tuple(fp.shape), (4096,))
        self.assertEqual(int(torch.count_nonzero(fp)), 0)
        self.assertTrue(torch.equal(starter.morgan_fingerprint('CCO'),
                                    starter.morgan_fingerprint('CCO', allow_unparseable=True)))

    def test_starter_accepts_valid_spectra_outside_its_preprocessing_window(self):
        import torch
        starter = load('starter_common', ROOT / 'environment/baseline/starter/common.py')
        for mzs, intensities in [([1500.0], [1.0]), ([50.0, 1500.0], [0.0, 1.0])]:
            with self.subTest(mzs=mzs):
                encoded = starter.spectrum_tensor(dict(
                    mzs=mzs, intensities=intensities, precursor_mz=1600.0))
                self.assertTrue(torch.isfinite(encoded).all())
                self.assertEqual(tuple(encoded.shape), (61, 2))
                self.assertEqual(float(encoded[0, 0]), 1600.0)
                self.assertEqual(int(torch.count_nonzero(encoded[1:])), 0)
        with self.assertRaisesRegex(ValueError, 'maximum intensity must be positive'):
            starter.spectrum_tensor(dict(mzs=[1500.0], intensities=[0.0], precursor_mz=1600.0))

    def test_declared_external_weights_are_not_authorized_by_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build(root, 'fixture')
            manifest = main_verifier.validate_submission(root)
            manifest['training_provenance']['pretrained_assets'] = [
                {'name': 'external', 'revision': 'immutable', 'license': 'MIT'}]
            (root / 'model_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'no authorized pretrained'):
                main_verifier.validate_submission(root)

    def test_failed_publication_preserves_previous_runnable_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'submission'
            publication.publish_directory(root, lambda stage: build(stage, 'first'))
            for path in [root, *root.rglob('*')]:
                if path.is_dir():
                    self.assertEqual(path.stat().st_mode & 0o005, 0o005)
                else:
                    self.assertEqual(path.stat().st_mode & 0o004, 0o004)
            before = (root / 'model_manifest.json').read_bytes()
            def interrupted(stage):
                build(stage, 'unfinished')
                raise RuntimeError('simulated preemption while exporting')
            with self.assertRaises(RuntimeError):
                publication.publish_directory(root, interrupted)
            self.assertEqual(before, (root / 'model_manifest.json').read_bytes())
            publication.publish_directory(root, lambda stage: build(stage, 'second'))
            relocated = Path(temporary) / 'elsewhere'
            shutil.copytree(root, relocated)
            shutil.rmtree(root)
            self.assertEqual(subprocess.check_output(['bash', str(relocated / 'infer.sh')], text=True).strip(), 'second')
            # A reader which captured the old manifest still has all its assets.
            old = json.loads(before)
            self.assertEqual((relocated / old['checkpoint']['path']).read_text(), 'first')
            compact = Path(temporary) / 'compact'
            publication.snapshot_package(relocated, compact)
            self.assertFalse((compact / old['checkpoint']['path']).exists())
            self.assertEqual(subprocess.check_output(['bash', str(compact / 'infer.sh')], text=True).strip(), 'second')

    def test_escaping_publication_is_rejected_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'submission'
            publication.publish_directory(root, lambda stage: build(stage, 'first'))
            before = (root / 'model_manifest.json').read_bytes()
            def escaping(stage):
                build(stage, 'bad')
                (stage / 'external').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                publication.publish_directory(root, escaping)
            self.assertEqual(before, (root / 'model_manifest.json').read_bytes())

    def test_cluster_bootstrap_keeps_spectrum_weighting(self):
        # One molecule has nine spectra, the other one: macro would be 0.5.
        report = diagnostics.cluster_interval([1] * 9 + [0], ['A'] * 9 + ['B'], resamples=500, seed=7)
        self.assertEqual(report['estimate'], 0.9)
        self.assertEqual(report['molecule_groups'], 2)
        self.assertEqual(report['lower'], 0)
        self.assertEqual(report['upper'], 1)
        self.assertEqual(report, diagnostics.cluster_interval([1] * 9 + [0], ['A'] * 9 + ['B'], resamples=500, seed=7))

    def test_no_difference_has_zero_paired_interval(self):
        report = diagnostics.cluster_interval([0, 0, 0], ['A', 'A', 'B'], resamples=100)
        self.assertEqual((report['estimate'], report['lower'], report['upper']), (0, 0, 0))

    def test_paired_report_resamples_molecules_without_changing_delta(self):
        truth = {f'q{i}': 'a' for i in range(10)}
        baseline = {q: 'b' for q in truth}
        challenger = {q: 'a' if q != 'q9' else 'b' for q in truth}
        groups = {q: 'A' if q != 'q9' else 'B' for q in truth}
        report = paired.compare(truth, baseline, challenger, confidence=0.95, resamples=500, seed=7, groups=groups)
        self.assertEqual(report['delta'], 0.9)
        self.assertEqual(report['molecule_groups'], 2)
        self.assertEqual(report['interval'], [0, 1])
        self.assertFalse(report['beats_reference'])

    def test_validation_rejects_nested_or_partial_rankings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'rankings.jsonl'
            for ranking in [[['a'], 'b'], ['a'], ['a', 'a']]:
                path.write_text(json.dumps({'query_id': 'q', 'ranking': ranking}) + '\n')
                with self.assertRaises(ValueError):
                    score.load_rankings(path, {'q': {'pool': 'p'}}, {'p': [{'candidate_id': 'a'}, {'candidate_id': 'b'}]})

    def test_routing_does_not_treat_missing_energy_as_zero(self):
        row = {'adduct': '[M+H]+', 'instrument_type': 'Orbitrap', 'collision_energy': None}
        self.assertFalse(diagnostics.released_domain(row))
        self.assertTrue(diagnostics.released_domain(dict(row, collision_energy=0.0)))
        self.assertFalse(diagnostics.released_domain(dict(row, collision_energy=float('nan'))))


if __name__ == '__main__':
    unittest.main()
