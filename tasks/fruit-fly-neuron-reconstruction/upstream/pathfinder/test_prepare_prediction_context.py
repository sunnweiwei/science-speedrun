import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from common import sha256
from test_stitch_ffn import fixture,xtemp_directory
from stitch_ffn import run as stitch
from pack_stitched import run as pack
from consensus_stitched import run as consensus
from prepare_prediction_context import run,audit
from domain_comparison import reconstruction_coverage
from infer_continuous_sense import validate_inputs
from infer_pipeline import validate_geometry
from shape_context import indexed_mesh,context_coverage
from context_storage import segmentation_path
from index_storage import load_index


def image_fixture(root):
    image=root/'image.json';np.save(root/'raw.npy',np.zeros((64,)*3,np.uint8))
    valid=np.ones(64,bool);valid[23]=False;np.save(root/'image_valid_z.npy',valid)
    image.write_text(json.dumps(dict(axes='zyx',shape=[64]*3,origin_nm=[0]*3,voxel_size_nm=[16]*3,
        coordinate_convention='voxel_outer_corner',files={name:sha256(root/name) for name in ('raw.npy','image_valid_z.npy')},
        image_valid=dict(scope='synthetic missing-plane fixture'))))
    return image


def prediction_fixture(root,image,index=0):
    folder=root/f'variant-{index}';folder.mkdir();m=fixture(folder)
    m.update(raw_sha256=sha256(image.parent/'raw.npy'),image_manifest=str(image),image_manifest_sha256=sha256(image),
        checkpoint_sha256=hashlib.sha256(f'checkpoint-{index//2}'.encode()).hexdigest(),seed_order='descending' if index%2 else 'ascending')
    for row in m['windows']:
        for key in ('raw_sha256','checkpoint_sha256','seed_order'):row[key]=m[key]
    # Exercise compatibility with already-frozen runs that omitted this field.
    m['metadata'].pop('raw_origin_nm')
    path=folder/'manifest.json';path.write_text(json.dumps(m));prediction=folder/'stitched'
    stitch(path,root,prediction,Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1)
    return prediction,path


@unittest.skipUnless(all(os.environ.get(k) for k in ('PATHFINDER_TEST_XTEMP','PATHFINDER_TEST_FFN_SOURCE','PATHFINDER_TEST_CONNECTOMICS_SOURCE')),'Xtemp and author sources required')
class ContextTests(unittest.TestCase):
    def check_consumers(self,image,output):
        r=json.loads((output/'receipt.json').read_text());i=load_index(output/'instance-index.json')
        raw=np.load(image.parent/'raw.npy');seg=np.load(segmentation_path(output));valid=np.load(image.parent/'image_valid_z.npy')
        self.assertEqual(validate_inputs(raw,seg,valid,json.loads(image.read_text()),r,i).tolist(),[20]*3)
        validate_geometry(raw,seg,r)
        self.assertTrue(reconstruction_coverage(output)['full_context_reconstruction_complete'])
        self.assertEqual(r['image_observation']['unknown_output_z'],[3])
        self.assertEqual(r['image_observation']['observed_z_fraction'],7/8)
        self.assertFalse(r['image_observation']['all_defects_detected'])
        coverage=json.loads((output/'coverage.json').read_text())
        for row in i['records']:
            vertices,faces,center=indexed_mesh(seg,row['id'],np.array([.016]*3),row)
            self.assertGreater(len(faces),0);self.assertTrue(np.isfinite(vertices).all())
            observed=context_coverage(center,np.array([.016]*3),r['metadata'],coverage)
            self.assertTrue(observed['prediction_context_complete'])
            self.assertLess(observed['image_observed_sphere_fraction'],1)

    def test_actual_stitched_result_consumed_without_relabeling_or_parent_edits(self):
        with xtemp_directory() as folder:
            root=Path(folder);image=image_fixture(root);pred,manifest=prediction_fixture(root,image)
            before=sha256(pred/'receipt.json');output=root/'context'
            import prepare_prediction_context as module
            with patch.object(module,'reserve',wraps=module.reserve) as reservation:
                result=run(pred,image,root,output,2,manifest,max_instances=100)
            self.assertGreater(reservation.call_args.args[1],(pred/'segmentation.npy').stat().st_size)
            self.assertEqual(result['segmentation_sha256'],sha256(pred/'segmentation.npy'))
            self.assertEqual(before,sha256(pred/'receipt.json'))
            self.assertEqual((output/'segmentation.npy').stat().st_ino,(pred/'segmentation.npy').stat().st_ino)
            self.check_consumers(image,output)
            # Retiring a redundant parent array does not invalidate its receipt.
            (pred/'segmentation.npy').unlink();self.assertTrue(audit(output)['full_context_reconstruction_complete'])
            cp=output/'coverage.json';c=json.loads(cp.read_text());c['tiles'][0]['stop'][0]-=1;cp.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'artifact changed'):audit(output)

    def test_numeric_context_matches_legacy_and_binds_table_bytes(self):
        with xtemp_directory() as folder:
            root=Path(folder);image=image_fixture(root);pred,manifest=prediction_fixture(root,image)
            run(pred,image,root,root/'legacy',2,manifest,max_instances=100,reference_parent_array=True)
            output=root/'numeric';run(pred,image,root,output,2,manifest,max_instances=100,reference_parent_array=True,numeric_max_label=100)
            legacy=load_index(root/'legacy/instance-index.json');numeric=load_index(output/'instance-index.json')
            self.assertEqual(legacy['records'],list(numeric['records']));numeric['records'].table._mmap.close()
            self.assertTrue(audit(output)['full_context_reconstruction_complete']);self.check_consumers(image,output)
            with (output/'instance-table.npy').open('ab') as stream:stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'changed'):audit(output)

    def test_parent_reference_avoids_duplicate_array_and_preserves_consumer_contract(self):
        with xtemp_directory() as folder:
            root=Path(folder);image=image_fixture(root);pred,manifest=prediction_fixture(root,image)
            before={str(p):sha256(p) for p in pred.iterdir() if p.is_file()}
            old=run(pred,image,root,root/'linked',2,manifest,max_instances=100)
            output=root/'referenced';result=run(pred,image,root,output,2,manifest,max_instances=100,reference_parent_array=True)
            self.assertFalse((output/'segmentation.npy').exists())
            self.assertEqual(segmentation_path(output),pred/'segmentation.npy')
            self.assertEqual(old['reserved_output_bytes']-result['reserved_output_bytes'],(pred/'segmentation.npy').stat().st_size)
            self.check_consumers(image,output)
            for path,digest in before.items():self.assertEqual(sha256(path),digest)
            receipt=output/'receipt.json';bad=dict(result,segmentation_path=str(root/'other.npy'));receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'Unambiguous'):audit(output)
            receipt.write_text(json.dumps(result));(output/'segmentation.npy').symlink_to(pred/'segmentation.npy')
            with self.assertRaisesRegex(ValueError,'Unambiguous'):audit(output)

    def test_six_real_author_stitches_and_global_consensus_use_same_downstream_contract(self):
        with xtemp_directory() as folder:
            root=Path(folder);image=image_fixture(root);packages=[]
            for i in range(6):
                pred,manifest=prediction_fixture(root,image,i);package=pred.parent/'package'
                pack(pred,manifest,root,package,100000,2);packages.append(str(package))
            spec=root/'spec.json';spec.write_text(json.dumps(packages));pred=root/'consensus'
            consensus(spec,root,pred,0,2,2,1000)
            run(pred,image,root,root/'context',2,max_instances=100)
            self.check_consumers(image,root/'context')
            archive=Path(packages[0])/'segmentation.npz'
            with archive.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'integrity'):audit(root/'context')

    def test_incomplete_changed_image_and_resource_limits_never_emit_complete_context(self):
        with xtemp_directory() as folder:
            root=Path(folder);image=image_fixture(root);pred,manifest=prediction_fixture(root,image)
            for kwargs,pattern in [({'memory_gib':.001,'max_instances':100},'memory'),({'memory_gib':2,'max_instances':1},'Instance limit')]:
                with self.assertRaisesRegex(ValueError,pattern):run(pred,image,root,root/'bad',manifest=manifest,**kwargs)
                self.assertFalse((root/'bad').exists())
            alternate=root/'different-image.json';j=json.loads(image.read_text());j['image_valid']['scope']='changed scope';alternate.write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,'Image manifest differs'):run(pred,alternate,root,root/'bad',2,manifest,max_instances=100)
            rp=pred/'receipt.json';r=json.loads(rp.read_text());r['complete']=False;rp.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Complete'):run(pred,image,root,root/'bad',2,manifest,max_instances=100)
            r['complete']=True;rp.write_text(json.dumps(r))
            with (root/'image_valid_z.npy').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Image artifact'):run(pred,image,root,root/'bad',2,manifest,max_instances=100)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
