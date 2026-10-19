import unittest
from unittest.mock import patch
from collections import deque
import numpy as np
from ffn_movement import next_move,disconnected_bias
from ffn_resegmentation import grow
from batched_ffn import grow_tile

class MovementTests(unittest.TestCase):
    def prediction(self,step,raw,canvas):
        logits=np.full_like(canvas,-4.)
        if step==1:
            logits[...,16,16,16]=4.
            logits[...,24,16,16]=4.
            logits[...,16,24,16]=4.
        else:
            # First move is +Z: keep original seed and current position,
            # but retract queued +Y (local [8,24,16]).
            logits[...,8,16,16]=4.
            logits[...,16,16,16]=4.
        return logits

    def test_resegmentation_skips_retracted_queued_position(self):
        calls=[]
        def predict(raw,seed):
            calls.append(1);return self.prediction(len(calls),raw,seed)
        _,report=grow(np.zeros((65,)*3,np.uint8),[32]*3,predict,max_calls=2)
        self.assertEqual(report['calls'],2)
        self.assertTrue(report['complete'])
        self.assertEqual(report['stale_moves_skipped'],1)

    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([[32]*3]))
    def test_tiled_segmentation_has_same_queue_safety(self):
        image=np.zeros((64,)*3,np.uint8);image[31:34,31:34,31:34]=200
        generator=grow_tile(image,[16]*3,[48]*3,max_calls=2);request=next(generator);calls=0
        while True:
            calls+=1
            try:request=generator.send(self.prediction(calls,*request))
            except StopIteration as result:
                _,report=result.value;break
        self.assertEqual(calls,2);self.assertEqual(report['seed_count'],1)
        self.assertFalse(report['budget_exhausted'])
        self.assertEqual(report['stale_moves_skipped'],1)

    def test_seed_retraction_terminates_even_with_positive_neighbor(self):
        def reject_seed(raw,seed):
            logits=np.full_like(seed,-4.);logits[...,24,16,16]=4.;return logits
        _,report=grow(np.zeros((65,)*3,np.uint8),[32]*3,reject_seed,max_calls=1)
        self.assertTrue(report['complete']);self.assertEqual(report['calls'],1)
        self.assertEqual(report['seeds_lost'],1)

    @patch('batched_ffn.policy_peaks',new=lambda image:np.asarray([[32]*3]))
    def test_rejected_seed_cannot_publish_disconnected_positive_voxels(self):
        image=np.zeros((64,)*3,np.uint8);image[31:34,31:34,31:34]=200
        generator=grow_tile(image,[16]*3,[48]*3);raw,canvas=next(generator)
        prediction=np.full_like(canvas,-4.)
        # Enough positive voxels to pass minimum object size, but no valid moves.
        prediction[17:20,17:20,17:20]=4.
        with self.assertRaises(StopIteration) as finished:generator.send(prediction)
        segmentation,report=finished.exception.value
        self.assertEqual(report['calls'],1)
        self.assertEqual(report['objects'],0)
        self.assertFalse(segmentation.any())

    def test_skipped_position_can_be_reenqueued_after_recovery(self):
        canvas=np.zeros((3,)*3);canvas[1,1,1]=4;queue=deque([(1,1,2)]);visited=set()
        self.assertIsNone(next_move(queue,visited,canvas,[1,1,1]))
        canvas[1,1,2]=4;queue.append((1,1,2))
        self.assertEqual(next_move(queue,visited,canvas,[1,1,1]),(1,1,2))

class DisconnectedBiasTests(unittest.TestCase):
    def test_unknown_padding_is_not_frozen_but_previous_rejection_is(self):
        old=np.array([-3.,-3.,3.,-3.]);new=np.array([4.,4.,-4.,-5.]);known=np.array([False,True,True,True])
        result=disconnected_bias(old,new,known)
        np.testing.assert_array_equal(result,[4.,-3.,-4.,-5.])
        np.testing.assert_array_equal(new,[4.,4.,-4.,-5.])

    def test_author_fraction_gate_is_strict_and_negative_disables(self):
        old=np.array([-3.,-3.]);new=np.array([4.,-4.]);known=np.ones(2,bool)
        np.testing.assert_array_equal(disconnected_bias(old,new,known,.5),new)
        np.testing.assert_array_equal(disconnected_bias(old,new,known,-1),new)
        np.testing.assert_array_equal(disconnected_bias(old,new,known,0),[-3.,-4.])


class FaceMaxTests(unittest.TestCase):
    def test_off_axis_face_peak_is_followed(self):
        from ffn_movement import face_max_offsets
        logits=np.full((33,)*3,-4.);logits[24,20,16]=4.
        self.assertEqual(face_max_offsets(logits),[(4.,(8,4,0))])

    def test_face_corner_deduplicates_and_sorts_scores(self):
        from ffn_movement import face_max_offsets
        logits=np.full((33,)*3,-4.);logits[24,24,24]=4.;logits[8,16,16]=5.
        self.assertEqual(face_max_offsets(logits),[(5.,(-8,0,0)),(4.,(8,8,8))])

    def test_seed_relative_cells_and_stale_checks(self):
        from ffn_movement import FaceMaxMovement
        movement=FaceMaxMovement([32]*3,[65]*3);canvas=np.full((65,)*3,4.)
        self.assertEqual(movement.next(canvas),(32,32,32))
        movement.visited.add(movement.key((32,32,32)))
        movement.queue.extend([(33,32,32),(40,36,32),(32,40,32)])
        canvas[40,36,32]=-4.;audit={}
        self.assertEqual(movement.next(canvas,audit),(32,40,32))
        self.assertEqual(audit['stale_moves_skipped'],1)
        self.assertEqual(movement.key((28,32,32)),(0,0,0))
        self.assertEqual(movement.key((27,32,32)),(-1,0,0))


    def test_occupied_center_is_skipped_without_marking_cell_visited(self):
        from ffn_movement import FaceMaxMovement
        movement=FaceMaxMovement([32]*3,[65]*3);canvas=np.full((65,)*3,4.)
        movement.queue.clear();movement.queue.extend([(40,32,32),(40,33,32)])
        audit={}
        self.assertEqual(movement.next(canvas,audit,blocked=lambda p:p==(40,32,32)),(40,33,32))
        self.assertEqual(audit['occupied_moves_skipped'],1)
        self.assertFalse(movement.visited)


    def test_resegmentation_does_not_move_into_third_party_object(self):
        occupied=np.zeros((65,)*3,bool);occupied[40,32,32]=True
        def predict(raw,canvas):
            logits=np.full_like(canvas,-4.);logits[...,16,16,16]=4.;logits[...,24,16,16]=4.
            return logits
        _,report=grow(np.zeros((65,)*3,np.uint8),[32]*3,predict,max_calls=1,occupied=occupied)
        self.assertTrue(report['complete'])
        self.assertEqual(report['calls'],1)
        self.assertEqual(report['occupied_moves_skipped'],1)


if __name__=='__main__':unittest.main()
