import json,tempfile,unittest
from pathlib import Path
from evaluate_physical import grid_control

# Stub with the funlib signatures: ERL = total length of edges whose two nodes share a predicted label.
STUB='''
def get_skeleton_lengths(graph,position_attributes,skeleton_id_attribute,store_edge_length=None):
    for u,v in graph.edges:graph.edges[u,v][store_edge_length]=1.0
    lengths={}
    for u,v in graph.edges:
        s=graph.nodes[u][skeleton_id_attribute];lengths[s]=lengths.get(s,0)+1.0
    return lengths
def expected_run_length(graph,skeleton_id_attribute,edge_length_attribute,node_segment_lut,skeleton_lengths=None):
    return float(sum(1.0 for u,v in graph.edges if node_segment_lut[u]==node_segment_lut[v]))
'''


class GridControlTests(unittest.TestCase):
    def test_best_cube_size_is_selected_from_the_fixed_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite=Path(tmp)
            meta={'axes':'zyx','coordinate_convention':'voxel_outer_corner','origin_nm':[0,0,0],'voxel_size_nm':[16,16,16],
                  'shape':[128,128,128],'scored_core_origin_nm':[512,512,512],'scored_core_shape':[64,64,64]}
            # One straight skeleton along x inside the core, one node per voxel.
            nodes=[{'id':i+1,'skeleton_id':1,'position_nm':[1000.,1000.,520.+16*i]} for i in range(60)]
            edges=[[i+1,i+2] for i in range(59)]
            (suite/'metadata.json').write_text(json.dumps(meta))
            (suite/'skeletons.json').write_text(json.dumps({'axes':'zyx','coordinate_units':'nm','nodes':nodes,'edges':edges}))
            (suite/'run_length.py').write_text(STUB)
            result=grid_control(suite,sizes=(4,8,32))
            # Bigger cubes cut the straight path fewer times, so the stub ERL grows with the cube size.
            self.assertEqual(result['cube_voxels'],32)
            self.assertGreater(result['nerl_by_size']['32'],result['nerl_by_size']['4'])


if __name__=='__main__':unittest.main()
