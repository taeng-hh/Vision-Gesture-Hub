import unittest,time
from pathlib import Path
from concurrent.futures import Future
import numpy as np
from ksl.predict import prediction_event
from core.game_cnn import GameCnnAssist,merge_assistance,build_rois

CLASSES=[dict(class_id='L',kind='word',text='왼쪽',game_action='left'),dict(class_id='R',kind='word',text='오른쪽',game_action='right'),dict(class_id='X',kind='word',text='친구',game_action=None)]
def pair(scores):
    return dict(probabilities=list(scores),normal=prediction_event(CLASSES,scores),game=prediction_event(CLASSES,scores,game_only=True))
def sample(t=0):
    f=np.zeros((67,3),np.float32);f[44]=[100,80,1];f[47]=[220,80,1]
    for offset in (0,21):
        for j in range(21):f[offset+j]=[110+j,120+j%4,1]
    image=np.zeros((240,320,3),np.uint8);image[:,:,0]=255
    return t,image,f

class CnnPolicyTests(unittest.TestCase):
    def test_only_agreeing_confident_rgb_can_rescue(self):
        result,status=merge_assistance(pair([.31,.13,.56]),[.95,.025,.025],CLASSES)
        self.assertTrue(result['game']['accepted']);self.assertEqual(result['game']['class_id'],'L')
        self.assertTrue(result['assistance_used'])
    def test_disagreement_or_weak_rgb_keeps_base(self):
        base=pair([.31,.13,.56])
        for rgb in [[.02,.96,.02],[.45,.25,.30]]:
            result,_=merge_assistance(base,rgb,CLASSES)
            self.assertIs(result,base);self.assertFalse(result['game']['accepted'])
    def test_rgb_cannot_rescue_tiny_keypoint_support(self):
        result,_=merge_assistance(pair([.01,.01,.98]),[.99,.005,.005],CLASSES)
        self.assertFalse(result['game']['accepted'])
    def test_roi_shape_color_and_motion(self):
        rois=build_rois([sample(i*.1) for i in range(5)])
        self.assertEqual(rois['images'].shape,(5,3,224,224,3))
        self.assertEqual(rois['motion'].shape,(5,8))
        self.assertEqual(rois['sample_positions'].shape,(64,))
        np.testing.assert_array_equal(rois['images'][0,0,112,112],[0,0,255])
    def test_base_mode_never_submits_cnn(self):
        a=GameCnnAssist(Path('.'),CLASSES)
        try:
            a._infer=lambda samples:self.fail('CNN should not run')
            p=pair([.31,.13,.56]);self.assertIs(a.evaluate(p,[],1,1),p)
            self.assertIsNone(a.pending)
        finally:a.close()
    def test_confident_base_does_not_wait(self):
        a=GameCnnAssist(Path('.'),CLASSES,True)
        try:
            a._infer=lambda samples:self.fail('Confident result should skip CNN')
            p=pair([.98,.01,.01]);self.assertIs(a.evaluate(p,[],1,1),p)
            self.assertIsNone(a.pending)
        finally:a.close()
    def test_mode_switch_discards_old_worker_result(self):
        a=GameCnnAssist(Path('.'),CLASSES,True)
        try:
            future=Future();a.pending=dict(future=future,generation=0,time=1,candidate='L',pair=pair([.31,.13,.56]))
            a.set_enabled(False);self.assertIsNone(a.pending);self.assertFalse(a.enabled)
            self.assertTrue(future.cancelled())
        finally:a.close()
    def test_old_or_different_candidate_result_cannot_fire(self):
        for now,current in [(3,pair([.31,.13,.56])),(1.2,pair([.13,.31,.56]))]:
            a=GameCnnAssist(Path('.'),CLASSES,True)
            try:
                f=Future();f.set_result(np.array([.95,.025,.025]))
                a.pending=dict(future=f,generation=0,time=1,candidate='L',pair=pair([.31,.13,.56]))
                self.assertIs(a.evaluate(current,[],now,1),current)
            finally:a.close()
    def test_worker_failure_falls_back_without_throwing(self):
        a=GameCnnAssist(Path('.'),CLASSES,True)
        try:
            f=Future();f.set_exception(RuntimeError('missing torchvision'))
            a.pending=dict(future=f,generation=0,time=1,candidate='L',pair=pair([.31,.13,.56]))
            p=pair([.31,.13,.56]);self.assertIs(a.evaluate(p,[],1.1,1),p)
            self.assertFalse(a.enabled);self.assertIn('missing torchvision',a.error)
        finally:a.close()
    def test_matching_result_uses_historical_input_probabilities(self):
        a=GameCnnAssist(Path('.'),CLASSES,True)
        try:
            old=pair([.31,.13,.56]);current=pair([.30,.15,.55]);f=Future();f.set_result(np.array([.95,.025,.025]))
            a.pending=dict(future=f,generation=0,time=1,candidate='L',pair=old)
            result=a.evaluate(current,[],1.1,1)
            self.assertTrue(result['game']['accepted'])
            self.assertAlmostEqual(result['game']['global_score'],.8*.31+.2*.95)
        finally:a.close()

if __name__=='__main__':unittest.main()
