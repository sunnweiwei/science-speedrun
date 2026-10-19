import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from test_fs_support import TemporaryDirectory
from model_package import construct,ffn_identity,verify_ffn_ensemble,ordered_ffn_runs,ENSEMBLE_PATHS,export,verify,stage as stage_package
from common import sha256


ARCH=dict(stage='ffn',width=32,residual_blocks=20,input_shape=[33]*3,voxel_size_nm=[16]*3)


def checkpoint(seed):
    torch.manual_seed(seed)
    return dict(architecture=copy.deepcopy(ARCH),model=construct('ffn',ARCH).state_dict(),
        step=seed,manifest_sha256='a'*64,validation_bce=.1)


def package(root,checkpoints):
    (root/'weights').mkdir(exist_ok=True);rows=[];files={}
    for name,c in zip(ENSEMBLE_PATHS,checkpoints):
        path=root/name;torch.save(c,path);files[name]=sha256(path)
        rows.append(dict(path=name,source_sha256=files[name],**ffn_identity(c)))
    return dict(files=files,models={'ffn':{'architecture':ARCH}},ffn_ensemble=dict(schema='three-checkpoint-ffn/v1',
        selection_rule='validation_bce',selection_verified=False,checkpoints=rows,ordered_runs=ordered_ffn_runs()))


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.checkpoints=[checkpoint(i) for i in (6750,7250,5750)]

    def test_different_training_unrolls_keep_pinned_metadata_and_same_inference_model(self):
        checkpoints=copy.deepcopy(self.checkpoints)
        for c,steps in zip(checkpoints,(4,8,4)):
            c['architecture']['ffn_recurrent_steps']=steps
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);manifest=package(root,checkpoints)
            result=verify_ffn_ensemble(root,manifest)
            self.assertEqual(result['variants'],6)
            self.assertEqual(manifest['ffn_ensemble']['checkpoints'][1]['architecture']['ffn_recurrent_steps'],8)

    def test_export_round_trip_and_legacy_package_cannot_satisfy_ensemble_gate(self):
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);inputs=root/'inputs';inputs.mkdir()
            paths={};members=[]
            for i,c in enumerate(self.checkpoints):
                path=inputs/f'ffn-{i}.pt';torch.save(dict(c,optimizer={'state':{}}),path);members.append(path)
            paths['ffn']=members[0]
            for stage,arch in [('sense',dict(stage='sense',width=8,factors=[1,2,4,8,8],input_shape=[128]*3,voxel_size_nm=[16]*3)),
                               ('shape',dict(stage='shape',points=8192,radius_um=10,widths=[64,128,256,512,1024,2048],neighbors=16))]:
                paths[stage]=inputs/f'{stage}.pt'
                torch.save(dict(architecture=arch,model=construct(stage,arch).state_dict(),step=1,validation_bce=.1,manifest_sha256='a'*64),paths[stage])
            native=['vendor/pointnext/openpoints/cpp/pointnet2_batch/src/'+n for n in ('sampling.cpp','sampling_gpu.cu','sampling_gpu.h','cuda_utils.h')]
            native+=['vendor/pointnext/LICENSE','vendor/pointnext/openpoints/LICENSE','point-ops/pathfinder_fps.so']
            for name in native:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU contract fixture; not a usable native extension')
            # Test packaging/strict real model loading; native ABI loading has
            # separate real Slurm evidence and is deliberately not simulated as
            # numerical GPU execution here.
            with patch('model_package.FPS'):
                result=export(root,root/'ensemble',paths,.8,members,'validation_bce')
                export(root,root/'legacy',paths,.8)
            self.assertEqual(result['ffn_ensemble']['variants'],6)
            budget=root/'new-budget';budget.mkdir();(budget/'budget.json').write_text('{}')
            (root/'ensemble/cached-prediction.npy').write_bytes(b'not an accepted model asset')
            with patch('model_package.FPS',side_effect=AssertionError('Staging must not load submitted native code')):
                staged=stage_package(budget,root/'ensemble',budget/'staged')
            self.assertEqual(staged['package_manifest_sha256'],result['package_manifest_sha256'])
            self.assertEqual(staged['copied_bytes'],sum(p.stat().st_size for p in (budget/'staged').rglob('*') if p.is_file()))
            self.assertFalse((budget/'staged/cached-prediction.npy').exists())
            for name in ENSEMBLE_PATHS:
                self.assertEqual(sha256(root/'ensemble'/name),sha256(budget/'staged'/name))
                self.assertNotEqual((root/'ensemble'/name).stat().st_ino,(budget/'staged'/name).stat().st_ino)
            with patch('model_package.reserve',side_effect=ValueError('200 GB storage limit would be exceeded')):
                with self.assertRaisesRegex(ValueError,'200 GB'):stage_package(budget,root/'ensemble',budget/'over-budget')
            self.assertFalse((budget/'over-budget').exists())
            with self.assertRaisesRegex(ValueError,'ensemble required'):stage_package(budget,root/'legacy',budget/'legacy')
            with patch('fetch_axonem.usage',return_value=0):
                single=stage_package(budget,root/'legacy',budget/'single',require_ensemble=False)
            self.assertNotIn('ffn_ensemble',single['check'])
            self.assertEqual(single['package_manifest_sha256'],sha256(root/'legacy/package.json'))
            with self.assertRaisesRegex(ValueError,'single-checkpoint'):
                stage_package(budget,root/'ensemble',budget/'wrong-protocol',require_ensemble=False)
            manifest_path=root/'ensemble/package.json';original_manifest=manifest_path.read_bytes()
            declared_prediction=json.loads(original_manifest)
            declared_prediction['files']['cached-prediction.npy']=sha256(root/'ensemble/cached-prediction.npy')
            manifest_path.write_text(json.dumps(declared_prediction))
            try:
                with self.assertRaisesRegex(ValueError,'Unrecognized model asset'):
                    stage_package(budget,root/'ensemble',budget/'predictions-not-models')
                self.assertFalse((budget/'predictions-not-models').exists())
            finally:manifest_path.write_bytes(original_manifest)
            from consensus_stitched import bind_model_package
            predictions=[dict(checkpoint_sha256=r['checkpoint_sha256'],seed_order=r['seed_order'])
                         for r in result['ffn_ensemble']['ordered_runs']]
            bound,pins=bind_model_package(root/'ensemble',predictions,root)
            self.assertTrue(bound['loaded_model_states_distinct'])
            self.assertEqual(bound['variants'],6)
            for path,digest in pins.items():self.assertEqual(sha256(path),digest)
            invalid=[predictions[:-1],list(reversed(predictions))]
            wrong=copy.deepcopy(predictions);wrong[2]['checkpoint_sha256']='f'*64;invalid.append(wrong)
            for rows in invalid:
                with self.assertRaisesRegex(ValueError,'ordered model-package'):
                    bind_model_package(root/'ensemble',rows,root)
            with self.assertRaisesRegex(ValueError,'ensemble required'):
                bind_model_package(root/'legacy',predictions,root)
            from merge_ffn_agglomeration import main as merge_main
            decisions=root/'decisions.json';decisions.write_text(json.dumps(dict(base_sha256='b',decisions=[dict(pair=[1,2])])) )
            shards=[]
            for i,row in enumerate(predictions[::2]):
                path=root/f'shard-{i}.json';shards.append(path)
                path.write_text(json.dumps(dict(shards=1,shard_index=0,checkpoint_sha256=row['checkpoint_sha256'],
                    raw_sha256='r',base_sha256='b',decisions_sha256=sha256(decisions),ground_truth_used=False,
                    results=[dict(pair=[1,2],complete=True,accepted=True,iou=.95,consistency=[[.95]*2]*2,
                                  prediction_voxels=[100,100],deleted_voxels=[0,0])])))
            args=['merge','--ensemble','--rule','pathfinder2025','--model-package',str(root/'ensemble'),
                  '--decisions',str(decisions),'--shards',*map(str,shards),'--output',str(root/'merged.json')]
            with patch('sys.argv',args):merge_main()
            merged=json.loads((root/'merged.json').read_text())
            self.assertEqual(merged['mapping'],{'2':1})
            self.assertEqual(merged['model_package_check']['package_manifest_sha256'],bound['package_manifest_sha256'])
            bad=json.loads(shards[0].read_text());bad['checkpoint_sha256']='f'*64;shards[0].write_text(json.dumps(bad))
            with patch('sys.argv',args[:-1]+[str(root/'bad-merged.json')]):
                with self.assertRaisesRegex(ValueError,'verified model package'):merge_main()
            self.assertFalse((root/'bad-merged.json').exists())
            self.assertEqual(verify(root/'ensemble',require_ensemble=True)['search_threshold'],.8)
            self.assertNotIn('ffn_ensemble',verify(root/'legacy'))
            with self.assertRaisesRegex(ValueError,'ensemble required'):verify(root/'legacy',require_ensemble=True)
            for name in ENSEMBLE_PATHS:self.assertNotIn('optimizer',torch.load(root/'ensemble'/name,weights_only=True))
            with self.assertRaisesRegex(ValueError,'First ensemble'):
                export(root,root/'wrong-first',paths,.8,[members[1],members[0],members[2]],'validation_bce')
            self.assertFalse((root/'wrong-first').exists())

    def test_real_full_ffn_assets_remain_valid_after_relocation(self):
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);old=root/'first';old.mkdir();manifest=package(old,self.checkpoints)
            before=verify_ffn_ensemble(old,manifest)
            new=root/'relocated';old.rename(new);after=verify_ffn_ensemble(new,manifest)
            self.assertTrue(after['loaded_model_states_distinct']);self.assertEqual(after['variants'],6)
            self.assertFalse(after['selection_verified'])
            self.assertEqual([(r['checkpoint_index'],r['seed_order']) for r in after['ordered_runs']],
                [(i,order) for i in range(3) for order in ('ascending','descending')])
            self.assertEqual([r['checkpoint_sha256'] for r in before['ordered_runs']],[r['checkpoint_sha256'] for r in after['ordered_runs']])
            self.assertTrue(all(Path(r['checkpoint_path']).parent==new/'weights' for r in after['ordered_runs']))

    def test_metadata_or_serialization_precision_cannot_fake_distinct_models(self):
        duplicate=copy.deepcopy(self.checkpoints[0]);duplicate['step']=9999
        duplicate['model']={k:v.double() for k,v in duplicate['model'].items()}
        self.assertEqual(ffn_identity(duplicate)['tensor_sha256'],ffn_identity(self.checkpoints[0])['tensor_sha256'])
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);manifest=package(root,[self.checkpoints[0],duplicate,self.checkpoints[2]])
            self.assertNotEqual(manifest['files'][ENSEMBLE_PATHS[0]],manifest['files'][ENSEMBLE_PATHS[1]])
            with self.assertRaisesRegex(ValueError,'distinct'):verify_ffn_ensemble(root,manifest)

    def test_reject_incomplete_order_unpinned_asset_and_false_selection_claim(self):
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);m=package(root,self.checkpoints)
            bad=[]
            x=copy.deepcopy(m);x['ffn_ensemble']['ordered_runs'].pop();bad.append(x)
            x=copy.deepcopy(m);x['ffn_ensemble']['ordered_runs'].reverse();bad.append(x)
            x=copy.deepcopy(m);x['ffn_ensemble']['checkpoints'].pop();bad.append(x)
            x=copy.deepcopy(m);x['ffn_ensemble']['selection_verified']=True;bad.append(x)
            x=copy.deepcopy(m);x['files'].pop(ENSEMBLE_PATHS[1]);bad.append(x)
            x=copy.deepcopy(m);x['ffn_ensemble']['checkpoints'][1]['path']='../outside.pt';bad.append(x)
            for manifest in bad:
                with self.assertRaises(ValueError):verify_ffn_ensemble(root,manifest)
            with (root/ENSEMBLE_PATHS[2]).open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify_ffn_ensemble(root,m)

    def test_training_origin_and_full_architecture_are_required(self):
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);checkpoints=copy.deepcopy(self.checkpoints);checkpoints[1]['manifest_sha256']='b'*64
            m=package(root,checkpoints)
            with self.assertRaisesRegex(ValueError,'training provenance'):verify_ffn_ensemble(root,m)
            for c in checkpoints:c.pop('manifest_sha256')
            m=package(root,checkpoints)
            with self.assertRaisesRegex(ValueError,'training manifest'):verify_ffn_ensemble(root,m)
        bad=copy.deepcopy(self.checkpoints[0]);bad['architecture']['width']=16
        with self.assertRaisesRegex(ValueError,'Full FFN'):ffn_identity(bad)
        bad=copy.deepcopy(self.checkpoints[0]);next(iter(bad['model'].values())).view(-1)[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Finite'):ffn_identity(bad)


if __name__=='__main__':unittest.main()
