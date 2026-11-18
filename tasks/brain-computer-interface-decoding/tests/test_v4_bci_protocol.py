"""CPU unit tests for SSI Benchmark V4 BCI protocol, de-contamination, isolation, and V3 metric parity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())
sys.path.insert(0, str(TASK_ROOT / 'environment/evaluation'))
sys.path.insert(0, str(TASK_ROOT / 'environment/baseline/braindecode'))
sys.path.insert(0, str(TASK_ROOT / 'tests'))

import bci_score
import main_verifier
import pretrained
import validate_pretrained


def _load_ssi_index_v3():
    spec = importlib.util.spec_from_file_location(
        'ssi_index_v3', REPO_ROOT / 'harness/ssi_index_v3.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSI_V3 = _load_ssi_index_v3()


def _stage_track(private_root: Path, track: str, records: list[tuple[dict, np.ndarray]]) -> None:
    """Pack all records into a single multi-subject shard (mimicking global-random-v2)."""
    tdir = private_root / track
    pub = tdir / 'public'
    pub.mkdir(parents=True, exist_ok=True)
    arr = np.stack([x for _, x in records]).astype(np.float32)
    np.save(pub / 'shard-000000.npy', arr, allow_pickle=False)
    truth_rows = []
    manifest_samples = []
    for idx, (meta, _) in enumerate(records):
        truth_rows.append(meta)
        manifest_samples.append({'id': meta['id'], 'path': 'shard-000000.npy', 'index': idx})
    (tdir / 'truth.jsonl').write_text('\n'.join(json.dumps(r) for r in truth_rows) + '\n')
    manifest = {
        'schema': 'bci-input-v1',
        'track': track,
        'config': {'channels': int(arr.shape[1]), 'seconds': int(arr.shape[2]), 'srate': int(arr.shape[3])},
        'samples': manifest_samples,
    }
    (pub / 'manifest.json').write_text(json.dumps(manifest))
    checksums = {
        'manifest.json': hashlib.sha256((pub / 'manifest.json').read_bytes()).hexdigest(),
        'shard-000000.npy': hashlib.sha256((pub / 'shard-000000.npy').read_bytes()).hexdigest(),
    }
    (pub / 'checksums.json').write_text(json.dumps(checksums))


def _make_window(c: int, s: int, ch0_val: float, ch1_val: float) -> np.ndarray:
    x = np.zeros((c, s, 200), dtype=np.float32)
    x[0, :, :] = ch0_val
    x[1, :, :] = ch1_val
    return x


class V4BciProtocolTests(unittest.TestCase):
    def test_faced_completely_removed(self):
        self.assertEqual(set(bci_score.CLASSES), {'tuev', 'ad65', 'openbmi', 'longitudinal'})
        self.assertEqual(bci_score.AGGREGATE_TRACKS, ('tuev', 'ad65', 'openbmi', 'longitudinal'))
        self.assertIn('faced', bci_score.DEPRECATED_TRACKS)

        for rel in (
            'instruction.md',
            'environment/contract.md',
            'environment/docker-compose.yaml',
            'environment/data/README.md',
        ):
            text = (TASK_ROOT / rel).read_text()
            self.assertNotIn('faced', text.lower(), f'deprecated FACED still referenced in {rel}')

        with self.assertRaises(ValueError):
            bci_score.score(
                'faced',
                [{'id': 'a', 'label': 0, 'subject': 's1', 'session': '1'}],
                np.ones((1, 9)) / 9.0,
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                main_verifier.run_verification(
                    root / 'sub',
                    root / 'priv',
                    root / 'out',
                    tracks=['faced'],
                    enforce_root=False,
                )

    def test_biot_removed_and_blocked_across_loader_validator_and_verifier(self):
        lock = json.loads(
            (TASK_ROOT / 'environment/baseline/braindecode/pretrained.lock.json').read_text()
        )
        self.assertEqual(set(lock['models']), {'cbramod', 'codebrain'})
        self.assertIn('biot', lock['disallowed_models'])
        biot_sha = 'a218f4054ae29962c6fd9d23d94c446d354c4f53e805a6771c85749ccc20baaa'
        self.assertEqual(lock['disallowed_models']['biot']['sha256'], biot_sha)

        audit_md = (TASK_ROOT / 'environment/references/PRETRAINED_AUDIT.md').read_text()
        self.assertIn('DISALLOWED', audit_md)
        self.assertIn('TUEV', audit_md)

        dummy_model = types.SimpleNamespace(backbone=types.SimpleNamespace(state_dict=lambda: {}))
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'disallowed'):
                pretrained.load_backbone(dummy_model, 'biot', tpath)

            (tpath / 'bci-biot.safetensors').write_bytes(b'forbidden-biot-placeholder')
            with self.assertRaisesRegex(ValueError, 'disallowed'):
                validate_pretrained.validate_directory(tpath)

        # Verifier metadata & filename enforcement
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            sub_meta = base / 'sub_meta'
            sub_meta.mkdir()
            (sub_meta / 'infer.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
            (sub_meta / 'config.json').write_text(json.dumps({'family': 'biot'}))
            with self.assertRaisesRegex(ValueError, 'BIOT'):
                main_verifier.validate_pretrained_policy(sub_meta)

            sub_init = base / 'sub_init'
            sub_init.mkdir()
            (sub_init / 'infer.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
            (sub_init / 'training.json').write_text(
                json.dumps(
                    {
                        'family': 'cbramod',
                        'initialization': {
                            'initialization': 'pretrained',
                            'repo': 'braindecode/biot-pretrained-six-datasets-18chs',
                            'sha256': biot_sha,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, 'BIOT'):
                main_verifier.validate_pretrained_policy(sub_init)

            sub_unauth = base / 'sub_unauth'
            sub_unauth.mkdir()
            (sub_unauth / 'infer.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
            (sub_unauth / 'training.json').write_text(
                json.dumps(
                    {
                        'family': 'cbramod',
                        'initialization': {
                            'initialization': 'pretrained',
                            'repo': 'custom/unverified-eeg',
                            'sha256': '0' * 64,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, 'unauthorized pretrained'):
                main_verifier.validate_pretrained_policy(sub_unauth)

            sub_file = base / 'sub_file'
            sub_file.mkdir()
            (sub_file / 'infer.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
            (sub_file / 'bci-biot.safetensors').write_bytes(b'1234')
            with self.assertRaisesRegex(ValueError, 'bci-biot.safetensors'):
                main_verifier.validate_pretrained_policy(sub_file)

    def test_per_subject_session_isolated_inference_blocks_transductive_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priv = root / 'private'
            temp_work = root / 'work'
            temp_work.mkdir()

            # Stage AD65: 3 held-out patients (p0->0, p1->1, p2->2), 4 windows each.
            # Channel 0 = pure patient-constant DC offset (-10, 0, +10), constant WITHIN each patient.
            # Channel 1 = true sample-local feature (0, 1, 2).
            ad65_records = []
            for label, (subj, dc_offset) in enumerate([('p0', -10.0), ('p1', 0.0), ('p2', 10.0)]):
                for w in range(4):
                    ad65_records.append(
                        (
                            {'id': f'ad65-{subj}-{w}', 'label': label, 'subject': subj, 'session': '1'},
                            _make_window(19, 10, ch0_val=dc_offset, ch1_val=float(label)),
                        )
                    )
            _stage_track(priv, 'ad65', ad65_records)

            # Stage TUEV with official-test subject and 6 distinct (patient, session) source stems
            tuev_records = []
            for cls in range(6):
                for w in range(2):
                    tuev_records.append(
                        (
                            {
                                'id': f'tuev-c{cls}-{w}',
                                'label': cls,
                                'subject': 'official-test',
                                'session': '1',
                                'source': f'processed_test/pat{cls:03d}_ses{cls:02d}_000-{w}.pkl',
                            },
                            _make_window(16, 5, ch0_val=float(cls * 5), ch1_val=float(cls)),
                        )
                    )
            _stage_track(priv, 'tuev', tuev_records)

            # Stage OpenBMI: 2 subjects, each with class 0 & 1
            openbmi_records = []
            for subj_idx, subj in enumerate(['s0', 's1']):
                for cls in (0, 1):
                    openbmi_records.append(
                        (
                            {'id': f'obmi-{subj}-{cls}', 'label': cls, 'subject': subj, 'session': '1'},
                            _make_window(62, 4, ch0_val=float(subj_idx * 10), ch1_val=float(cls)),
                        )
                    )
            _stage_track(priv, 'openbmi', openbmi_records)

            # Stage Longitudinal: 2 subjects x 2 sessions, each session has eyes closed (0) & open (1)
            long_records = []
            for subj_idx, subj in enumerate(['l0', 'l1']):
                for sess in ('1', '2'):
                    for cls in (0, 1):
                        long_records.append(
                            (
                                {'id': f'long-{subj}-s{sess}-{cls}', 'label': cls, 'subject': subj, 'session': sess},
                                _make_window(58, 2, ch0_val=float(subj_idx * 7), ch1_val=float(cls)),
                            )
                        )
            _stage_track(priv, 'longitudinal', long_records)

            # --- Canary A: Transductive cross-patient cohort-mean subtraction on AD65 ---
            canary_pkg = root / 'canary_transductive'
            canary_pkg.mkdir()
            (canary_pkg / 'canary_infer.py').write_text(
                textwrap.dedent(
                    """
                    import json, sys
                    from pathlib import Path
                    import numpy as np

                    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
                    manifest = json.loads((inp / 'manifest.json').read_text())
                    shards = {
                        name: np.load(inp / name, allow_pickle=False)
                        for name in {s['path'] for s in manifest['samples']}
                    }
                    # Transductive cheat: subtract cohort mean of channel 0 across ALL samples in INPUT_DIR
                    raw_ch0 = np.array([float(shards[s['path']][s['index'], 0].mean()) for s in manifest['samples']])
                    centered = raw_ch0 - float(raw_ch0.mean())
                    rows = []
                    for s, z in zip(manifest['samples'], centered):
                        if z < -2.0:
                            probs = [0.98, 0.01, 0.01]
                        elif z > 2.0:
                            probs = [0.01, 0.01, 0.98]
                        else:
                            probs = [0.01, 0.98, 0.01]
                        rows.append(json.dumps({'id': s['id'], 'probabilities': probs}))
                    (out / 'predictions.jsonl').write_text('\\n'.join(rows) + '\\n')
                    """
                ).strip()
                + '\n'
            )
            (canary_pkg / 'infer.sh').write_text(
                f'#!/usr/bin/env bash\nset -euo pipefail\n"{sys.executable}" "$(dirname "$0")/canary_infer.py" "$1" "$2"\n'
            )

            # 1) Verify Canary A WOULD cheat to 1.0 if given the un-isolated multi-patient shard
            unisolated_out = root / 'unisolated_out'
            unisolated_out.mkdir()
            subprocess.run(
                ['bash', str(canary_pkg / 'infer.sh'), str(priv / 'ad65/public'), str(unisolated_out)],
                check=True,
            )
            ad65_truth = [json.loads(l) for l in (priv / 'ad65/truth.jsonl').read_text().splitlines()]
            unisolated_probs = bci_score.read_predictions(unisolated_out / 'predictions.jsonl', ad65_truth, 3)
            unisolated_res = bci_score.score('ad65', ad65_truth, unisolated_probs)
            self.assertAlmostEqual(unisolated_res['reward'], 1.0, places=12)

            # 2) Under V4 isolated verifier, each patient runs alone -> centered == 0 -> collapses to chance (1/3)!
            isolated_res = main_verifier.evaluate(
                'ad65',
                canary_pkg,
                priv,
                limit=0,
                timeout=60,
                enforce_root=False,
                temp_root=temp_work,
            )
            self.assertEqual(isolated_res['status'], 'ok')
            self.assertEqual(isolated_res['inference_isolation'], 'per_subject_session')
            self.assertEqual(isolated_res['isolated_inference_units'], 3)
            self.assertAlmostEqual(isolated_res['reward'], 1.0 / 3.0, places=12)
            self.assertAlmostEqual(isolated_res['chance_adjusted_skill'], 0.0, places=12)

            # --- Compliant Model B: Sample-local decoder using within-window channel 1 ---
            local_pkg = root / 'compliant_local'
            local_pkg.mkdir()
            (local_pkg / 'local_infer.py').write_text(
                textwrap.dedent(
                    """
                    import json, sys
                    from pathlib import Path
                    import numpy as np

                    CLASSES = {'tuev': 6, 'ad65': 3, 'openbmi': 2, 'longitudinal': 2}
                    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
                    manifest = json.loads((inp / 'manifest.json').read_text())
                    k = CLASSES[manifest['track']]
                    shards = {
                        name: np.load(inp / name, allow_pickle=False)
                        for name in {s['path'] for s in manifest['samples']}
                    }
                    # Also assert physical shard isolation: shard-000000.npy never holds the whole cohort!
                    assert shards['shard-000000.npy'].shape[0] == len(manifest['samples'])
                    assert len(manifest['samples']) <= 4, f'unexpectedly un-isolated batch size: {len(manifest["samples"])}'
                    rows = []
                    for s in manifest['samples']:
                        cls = int(round(float(shards[s['path']][s['index'], 1].mean())))
                        probs = [0.0] * k
                        probs[cls] = 1.0
                        rows.append(json.dumps({'id': s['id'], 'probabilities': probs}))
                    (out / 'predictions.jsonl').write_text('\\n'.join(rows) + '\\n')
                    """
                ).strip()
                + '\n'
            )
            (local_pkg / 'infer.sh').write_text(
                f'#!/usr/bin/env bash\nset -euo pipefail\n"{sys.executable}" "$(dirname "$0")/local_infer.py" "$1" "$2"\n'
            )

            out_dir = root / 'logs_compliant'
            full_report = main_verifier.run_verification(
                local_pkg,
                priv,
                out_dir,
                enforce_root=False,
                temp_root=temp_work,
            )
            self.assertEqual(full_report['tracks']['tuev']['isolated_inference_units'], 6)
            self.assertEqual(full_report['tracks']['ad65']['isolated_inference_units'], 3)
            self.assertEqual(full_report['tracks']['openbmi']['isolated_inference_units'], 2)
            self.assertEqual(full_report['tracks']['longitudinal']['isolated_inference_units'], 4)

            reward_json = json.loads((out_dir / 'reward.json').read_text())
            self.assertAlmostEqual(reward_json['score'], 1.0, places=12)
            self.assertAlmostEqual(reward_json['reward'], 1.0, places=12)
            self.assertAlmostEqual(reward_json['score'], SSI_V3.bci(reward_json)['score'], places=12)

    def test_subject_session_stratified_ba_and_ssi_index_v3_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priv = root / 'private'
            temp_work = root / 'work'
            temp_work.mkdir()

            # Construct imbalanced window counts per subject so window-pooled BA != subject/session BA!
            # Channel 1 encodes the predicted class emitted by our deterministic mock model.
            # TUEV: 6 classes, 2 windows each; model gets classes 0,1,2 right and maps 3,4,5 -> 0 => BA = 3/6 = 0.5
            tuev_records = []
            for cls in range(6):
                pred_cls = cls if cls < 3 else 0
                for w in range(2):
                    tuev_records.append(
                        (
                            {
                                'id': f't-{cls}-{w}',
                                'label': cls,
                                'subject': 'official-test',
                                'session': '1',
                                'source': f'processed_test/p{cls}_s1_0-{w}.pkl',
                            },
                            _make_window(16, 5, 0.0, float(pred_cls)),
                        )
                    )
            _stage_track(priv, 'tuev', tuev_records)

            # AD65:
            # p0 (label 0): 20 windows all predicted 1 (wrong!)
            # p1 (label 0): 1 window predicted 0 (right!)
            # p2 (label 1): 1 window predicted 1 (right!)
            # p3 (label 2): 1 window predicted 2 (right!)
            # Patient-level BA: class 0 has 1/2 = 0.5, class 1 has 1.0, class 2 has 1.0 -> BA_subj = 5/6 = 0.833333333333
            # Window-pooled BA: class 0 has 1/21, class 1 has 1.0, class 2 has 1.0 -> BA_win = (1/21 + 2)/3 = 0.68253968254
            ad65_records = []
            for w in range(20):
                ad65_records.append(
                    ({'id': f'a-p0-{w}', 'label': 0, 'subject': 'p0', 'session': '1'}, _make_window(19, 10, 0.0, 1.0))
                )
            ad65_records.append(
                ({'id': 'a-p1-0', 'label': 0, 'subject': 'p1', 'session': '1'}, _make_window(19, 10, 0.0, 0.0))
            )
            ad65_records.append(
                ({'id': 'a-p2-0', 'label': 1, 'subject': 'p2', 'session': '1'}, _make_window(19, 10, 0.0, 1.0))
            )
            ad65_records.append(
                ({'id': 'a-p3-0', 'label': 2, 'subject': 'p3', 'session': '1'}, _make_window(19, 10, 0.0, 2.0))
            )
            _stage_track(priv, 'ad65', ad65_records)

            # OpenBMI:
            # s0: 20 trials (10 cls 0, 10 cls 1), all predicted 0 -> s0 BA = 0.5
            # s1: 2 trials (1 cls 0, 1 cls 1), both right -> s1 BA = 1.0
            # s2: 2 trials (1 cls 0, 1 cls 1), both right -> s2 BA = 1.0
            # Subject-stratified BA = (0.5 + 1.0 + 1.0) / 3 = 5/6 = 0.833333333333
            # Window-pooled BA = (12/12 + 2/12) / 2 = 7/12 = 0.583333333333
            obmi_records = []
            for w in range(10):
                obmi_records.append(
                    ({'id': f'o-s0-0-{w}', 'label': 0, 'subject': 's0', 'session': '1'}, _make_window(62, 4, 0.0, 0.0))
                )
                obmi_records.append(
                    ({'id': f'o-s0-1-{w}', 'label': 1, 'subject': 's0', 'session': '1'}, _make_window(62, 4, 0.0, 0.0))
                )
            for subj in ('s1', 's2'):
                for cls in (0, 1):
                    obmi_records.append(
                        (
                            {'id': f'o-{subj}-{cls}', 'label': cls, 'subject': subj, 'session': '1'},
                            _make_window(62, 4, 0.0, float(cls)),
                        )
                    )
            _stage_track(priv, 'openbmi', obmi_records)

            # Longitudinal:
            # l0 session 1: 20 windows (10 cls 0, 10 cls 1) all predicted 0 -> BA = 0.5
            # l0 session 2: 2 windows (1 cls 0, 1 cls 1) both right -> BA = 1.0 => l0 mean = 0.75
            # l1 session 1: 2 windows both right -> BA = 1.0 => l1 mean = 1.0
            # Subject/session-stratified BA = (0.75 + 1.0) / 2 = 0.875
            # Window-pooled BA = (12/12 + 2/12) / 2 = 7/12 = 0.583333333333
            long_records = []
            for w in range(10):
                long_records.append(
                    ({'id': f'l-l0-s1-0-{w}', 'label': 0, 'subject': 'l0', 'session': '1'}, _make_window(58, 2, 0.0, 0.0))
                )
                long_records.append(
                    ({'id': f'l-l0-s1-1-{w}', 'label': 1, 'subject': 'l0', 'session': '1'}, _make_window(58, 2, 0.0, 0.0))
                )
            for cls in (0, 1):
                long_records.append(
                    ({'id': f'l-l0-s2-{cls}', 'label': cls, 'subject': 'l0', 'session': '2'}, _make_window(58, 2, 0.0, float(cls)))
                )
                long_records.append(
                    ({'id': f'l-l1-s1-{cls}', 'label': cls, 'subject': 'l1', 'session': '1'}, _make_window(58, 2, 0.0, float(cls)))
                )
            _stage_track(priv, 'longitudinal', long_records)

            pkg = root / 'pkg'
            pkg.mkdir()
            (pkg / 'emit.py').write_text(
                textwrap.dedent(
                    """
                    import json, sys
                    from pathlib import Path
                    import numpy as np

                    CLASSES = {'tuev': 6, 'ad65': 3, 'openbmi': 2, 'longitudinal': 2}
                    inp, out = Path(sys.argv[1]), Path(sys.argv[2])
                    manifest = json.loads((inp / 'manifest.json').read_text())
                    k = CLASSES[manifest['track']]
                    shard = np.load(inp / 'shard-000000.npy', allow_pickle=False)
                    rows = []
                    for s in manifest['samples']:
                        cls = int(round(float(shard[s['index'], 1].mean())))
                        probs = [0.0] * k
                        probs[cls] = 1.0
                        rows.append(json.dumps({'id': s['id'], 'probabilities': probs}))
                    (out / 'predictions.jsonl').write_text('\\n'.join(rows) + '\\n')
                    """
                ).strip()
                + '\n'
            )
            (pkg / 'infer.sh').write_text(
                f'#!/usr/bin/env bash\nset -euo pipefail\n"{sys.executable}" "$(dirname "$0")/emit.py" "$1" "$2"\n'
            )

            out_dir = root / 'logs'
            report = main_verifier.run_verification(pkg, priv, out_dir, enforce_root=False, temp_root=temp_work)
            rew = json.loads((out_dir / 'reward.json').read_text())

            self.assertAlmostEqual(rew['tuev'], 0.5, places=12)
            self.assertAlmostEqual(rew['ad65'], 5.0 / 6.0, places=12)
            self.assertAlmostEqual(rew['openbmi'], 5.0 / 6.0, places=12)
            self.assertAlmostEqual(rew['longitudinal'], 0.875, places=12)

            # Confirm stratified BA != window-pooled BA on the 3 multi-window-per-subject tracks
            for trk in ('ad65', 'openbmi', 'longitudinal'):
                self.assertGreater(
                    abs(report['tracks'][trk]['primary_balanced_accuracy'] - report['tracks'][trk]['window_balanced_accuracy']),
                    0.10,
                )

            v3_ref = SSI_V3.bci(rew)
            self.assertLess(abs(rew['score'] - v3_ref['score']), 1e-12)
            self.assertLess(abs(rew['reward'] - v3_ref['score']), 1e-12)
            self.assertLess(abs(rew['signed_score'] - v3_ref['signed_score']), 1e-12)

            # Check additional boundary fixtures against ssi_index_v3.bci()
            fixtures = [
                {'tuev': 1.0 / 6.0, 'ad65': 1.0 / 3.0, 'openbmi': 0.5, 'longitudinal': 0.5},  # exact chance -> 0.0
                {'tuev': 0.0, 'ad65': 0.0, 'openbmi': 0.0, 'longitudinal': 0.0},  # all invalid/zero -> clipped 0.0
                {'tuev': 0.5577, 'ad65': 0.5975, 'openbmi': 0.6331, 'longitudinal': 0.7689},  # CodeBrain reference
                {'tuev': 0.0, 'ad65': 0.8, 'openbmi': 0.75, 'longitudinal': 0.9},  # partial failure (tuev failed=0)
                {'tuev': 1.0, 'ad65': 1.0, 'openbmi': 1.0, 'longitudinal': 1.0},  # perfect -> 1.0
            ]
            for fx in fixtures:
                fx_dir = root / 'fx'
                fx_dir.mkdir(exist_ok=True)
                mock_report = {
                    'scope': 'full',
                    'tracks': {k: {'reward': v} for k, v in fx.items()},
                }
                main_verifier.write_receipts(fx_dir, mock_report)
                emitted = json.loads((fx_dir / 'reward.json').read_text())
                ref = SSI_V3.bci(emitted)
                self.assertLess(abs(emitted['score'] - ref['score']), 1e-12)
                self.assertLess(abs(emitted['reward'] - ref['score']), 1e-12)
                self.assertLess(abs(emitted['signed_score'] - ref['signed_score']), 1e-12)

    def test_visibility_boundary_and_scientific_readme(self):
        dockerfile = (TASK_ROOT / 'environment/Dockerfile').read_text()
        compose = (TASK_ROOT / 'environment/docker-compose.yaml').read_text()
        self.assertNotIn('tests/', dockerfile)
        self.assertNotIn('tests/', compose)
        self.assertNotIn('TASK_PRIVATE_DIR', compose)

        tests_readme = (TASK_ROOT / 'tests/README.md').read_text()
        for heading in (
            '## Scientific & Literature Justification of Verifier Design',
            '### 1. Primary Metric & Upstream Paper Provenance',
            '### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted',
            '### 3. Diagnostic Signals Retained in report.json',
        ):
            self.assertIn(heading, tests_readme)
        for citation in ('LaBraM', 'EEGPT', 'CBraMod'):
            self.assertIn(citation, tests_readme)


if __name__ == '__main__':
    unittest.main()
