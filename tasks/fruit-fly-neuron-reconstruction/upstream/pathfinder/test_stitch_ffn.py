import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from stitch_ffn import windows,validate,run
from common import sha256
from window_io import save_compressed,reserved_window_bytes


@contextmanager
def xtemp_directory():
    from test_fs_support import TemporaryDirectory
    with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
        yield folder


def fixture(root,roi=None,boundary_image_shape=None):
    roi=[20,28]*3 if roi is None else roi;records=[];split=roi[2]+3
    for w in windows(roi,4,2,boundary_image_shape):
        coords=np.indices(tuple(np.asarray(w['stop'])-w['start']))+np.asarray(w['start'])[:,None,None,None]
        array=np.where(coords[1]<split,(w['index']+1)*10,(w['index']+1)*10+1).astype(np.uint32)
        array[coords[1]==split]=0
        path=root/f"window-{w['index']}.npy";np.save(path,array)
        records.append(dict(w,path=str(path),sha256=sha256(path),budget_exhausted=False,raw_sha256='synthetic-image',checkpoint_sha256='synthetic-model',seed_order='ascending'))
    image=root/'image.json';image.write_text(json.dumps(dict(axes='zyx',shape=boundary_image_shape or [64]*3,origin_nm=[0]*3,voxel_size_nm=[16]*3,coordinate_convention='voxel_outer_corner',files={'raw.npy':'synthetic-image'})))
    result=dict(schema='overlapping-ffn-windows/v1',ground_truth_used=False,roi=roi,core_size=4,margin=2,windows=records,raw_sha256='synthetic-image',checkpoint_sha256='synthetic-model',seed_order='ascending',image_manifest=str(image),image_manifest_sha256=sha256(image),metadata=dict(axes='zyx',shape=[8]*3,origin_nm=[v*16 for v in roi[::2]],raw_origin_nm=[0]*3,voxel_size_nm=[16]*3,coordinate_convention='voxel_outer_corner'))
    if boundary_image_shape is not None:result['boundary_image_shape']=boundary_image_shape
    return result


class StitchTests(unittest.TestCase):
    def test_full_frozen_domain_keeps_network_halo_without_cropping_cores(self):
        rows=windows([16,1778]*3,436,32,[1794]*3)
        self.assertEqual(len(rows),125)
        self.assertEqual(sum(int(np.prod(np.asarray(r['core_stop'])-r['core_start'])) for r in rows),1762**3)
        for r in rows:
            self.assertTrue((np.asarray(r['start'])>=16).all())
            self.assertTrue((np.asarray(r['stop'])<=1778).all())
            self.assertTrue((np.asarray(r['start'])<=r['core_start']).all())
            self.assertTrue((np.asarray(r['stop'])>=r['core_stop']).all())
        for axis in range(3):
            bounds=sorted({(r['core_start'][axis],r['core_stop'][axis]) for r in rows})
            self.assertEqual(bounds[0][0],16);self.assertEqual(bounds[-1][1],1778)
            self.assertTrue(all(a[1]==b[0] for a,b in zip(bounds,bounds[1:])))
        with self.assertRaises(ValueError):windows([15,1778]*3,436,32,[1794]*3)
        with self.assertRaises(ValueError):windows([16,1779]*3,436,32,[1794]*3)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP') and os.environ.get('PATHFINDER_TEST_FFN_SOURCE') and os.environ.get('PATHFINDER_TEST_CONNECTOMICS_SOURCE'),'Xtemp and author sources required')
    def test_actual_stitch_with_clipped_outer_envelopes(self):
        with xtemp_directory() as folder:
            root=Path(folder);manifest=fixture(root,[16,24]*3,[40]*3)
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            run(path,root,root/'output',Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1)
            actual=np.load(root/'output/segmentation.npy')
            self.assertEqual(actual.shape,(8,8,8));self.assertEqual(len(np.unique(actual)),3)
            self.assertEqual(int((actual>0).sum()),8*7*8)
            manifest['compressed_window_bytes']=100000
            for row in manifest['windows']:
                array=np.load(row['path'])
                storage=save_compressed(array,root/f"window-{row['index']}.npz",reserved_window_bytes(array.shape,100000))
                self.assertEqual(storage['sha256'],row['sha256'])
                row.update(storage)
            path.write_text(json.dumps(manifest))
            run(path,root,root/'compressed-output',Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1)
            np.testing.assert_array_equal(actual,np.load(root/'compressed-output/segmentation.npy'))
            core_result=run(path,root,root/'compressed-cores-output',Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),
                Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1,compressed_core_bytes=10000,compression_level=6,mapping_bytes=100000)
            self.assertEqual(sha256(root/'output/segmentation.npy'),core_result['segmentation_sha256'])
            original=json.loads((root/'output/receipt.json').read_text())
            self.assertEqual([r['sha256'] for r in original['cores']],[r['sha256'] for r in core_result['cores']])
            self.assertTrue(all(r['storage']=='npz' for r in core_result['cores']))
            self.assertFalse(list((root/'compressed-cores-output/cores').glob('*.npy')))
            self.assertEqual((root/'output/mapping.json').read_bytes(),(root/'compressed-cores-output/mapping.json').read_bytes())
            with self.assertRaisesRegex(ValueError,'byte cap'):
                run(path,root,root/'mapping-overflow',Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),
                    Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1,mapping_bytes=10)
            self.assertFalse((root/'mapping-overflow/receipt.json').exists())
            self.assertFalse((root/'mapping-overflow/mapping.json').exists())

    def test_geometry_tiles_core_exactly_and_preserves_overlap(self):
        ws=windows([50,59,50,58,50,57],4,2);cover=np.zeros((9,8,7),np.uint8)
        for w in ws:
            lo=np.asarray(w['core_start'])-50;hi=np.asarray(w['core_stop'])-50
            cover[tuple(slice(a,b) for a,b in zip(lo,hi))]+=1
            np.testing.assert_array_equal(np.asarray(w['core_start'])-w['start'],[2]*3)
        self.assertTrue((cover==1).all())

    def test_small_disconnected_pieces_are_cleared_and_relabelled(self):
        from stitch_ffn import drop_small_components
        core=np.zeros((4,6,6),np.uint64);core[0,0,0]=1;core[1:4,1:5,1:5]=2;core[0,5,5]=3;core[0,0,2:5]=4
        out,count,dropped=drop_small_components(core,min_voxels=3)
        self.assertEqual(count,2);self.assertEqual(dropped,(2,2))
        self.assertEqual(out[0,0,0],0);self.assertEqual(out[0,5,5],0)
        self.assertTrue((out[1:4,1:5,1:5]==1).all());self.assertTrue((out[0,0,2:5]==2).all())

    def test_partial_duplicate_and_mixed_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=fixture(Path(folder));self.assertEqual(len(validate(manifest)),8)
            cases=[]
            d=copy.deepcopy(manifest);d['windows'][0]['budget_exhausted']=True;cases.append(d)
            d=copy.deepcopy(manifest);d['windows'][0].update(budget_exhausted=True,call_limit_reached=False);cases.append(d)
            d=copy.deepcopy(manifest);d['windows'][-1]=d['windows'][0];cases.append(d)
            d=copy.deepcopy(manifest);d['windows'][0]['seed_order']='descending';cases.append(d)
            d=copy.deepcopy(manifest);d['windows'].pop();cases.append(d)
            for bad in cases:
                with self.assertRaises(ValueError):validate(bad)
            # Windows stopped by the per-window call cap keep their grown segments.
            d=copy.deepcopy(manifest);d['windows'][0].update(budget_exhausted=True,call_limit_reached=True)
            self.assertEqual(len(validate(d)),8)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_XTEMP') and os.environ.get('PATHFINDER_TEST_FFN_SOURCE') and os.environ.get('PATHFINDER_TEST_CONNECTOMICS_SOURCE'),'Xtemp and author sources required')
    def test_actual_author_components_and_global_stitch_preserve_two_objects(self):
        with xtemp_directory() as folder:
            root=Path(folder);manifest=fixture(root);path=root/'manifest.json';path.write_text(json.dumps(manifest))
            ffn=Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']);connect=Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE'])
            import stitch_ffn
            with patch('stitch_ffn.window_array',wraps=stitch_ffn.window_array) as reads:
                result=run(path,root,root/'output',ffn,connect,1)
            # Boundary matching must not repeatedly decode complete 3D windows.
            for row in manifest['windows']:
                self.assertLessEqual(sum(call.args[0]['path']==row['path'] for call in reads.call_args_list),2)
            planes=json.loads((root/'output/interface-planes.json').read_text())
            self.assertEqual(len(planes['planes']),2*result['interfaces'])
            self.assertEqual(planes['cache_bytes'],planes['cache_bound_bytes'])
            self.assertTrue(result['complete']);self.assertEqual(result['interfaces'],12)
            from infer_pipeline import validate_geometry
            validate_geometry(np.zeros((64,64,64),np.uint8),np.load(root/'output/segmentation.npy'),result)
            actual=np.load(root/'output/segmentation.npy')
            self.assertEqual(len(np.unique(actual)),3)
            self.assertTrue((actual[:,3,:]==0).all())
            self.assertEqual(len(np.unique(actual[:,:3,:])),1)
            self.assertEqual(len(np.unique(actual[:,4:,:])),1)
            self.assertNotEqual(actual[0,0,0],actual[0,4,0])
            # Relabeling/merging cannot change the foreground support.
            self.assertEqual(int((actual>0).sum()),8*7*8)
            manifest['metadata']['origin_nm'][0]+=16;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Physical output origin'):
                run(path,root,root/'bad-origin',ffn,connect,1)
            self.assertFalse((root/'bad-origin').exists())
            manifest['metadata']['origin_nm'][0]-=16;manifest['metadata']['raw_origin_nm'][0]=1;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Raw origin'):
                run(path,root,root/'bad-raw-origin',ffn,connect,1)
            self.assertFalse((root/'bad-raw-origin').exists())


if __name__=='__main__':unittest.main()
