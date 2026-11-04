"""Keypoint-first decisions: CNN may corroborate but never replace the candidate."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from fusion.predict import select_prediction,FusionRecognizer
from fusion.fast import analyze_clip,needs_mode_analysis
from tests.test_fusion_speed import clip_with_landmarks

CLASSES=[dict(class_id='A',kind='word',text='a'),dict(class_id='B',kind='word',text='b')]
def raw(kp,rgb=None):
    return dict(keypoint_probabilities=kp,rgb_probabilities=rgb,keypoint_weight=.1,
                computed_modalities=['keypoint']+(['rgb'] if rgb is not None else []),
                available_modalities=['keypoint'],component_errors={})
def select(kp,rgb=None):return select_prediction(raw(kp,rgb),CLASSES,'assisted')

class AssistedPolicyTests(unittest.TestCase):
    def test_confident_keypoint_is_never_overruled_by_cnn(self):
        r=select([.99,.01],[.01,.99])
        self.assertEqual(r['class_id'],'A');self.assertEqual(r['assist_status'],'keypoint_confident')
        self.assertEqual(r['decision_keypoint_weight'],1.)
    def test_cnn_can_confirm_same_borderline_candidate(self):
        r=select([.72,.28],[.99,.01])
        self.assertTrue(r['accepted']);self.assertEqual(r['class_id'],'A')
        self.assertAlmostEqual(r['score'],.774);self.assertTrue(r['assistance_used'])
    def test_disagreeing_cnn_cannot_replace_candidate(self):
        r=select([.72,.28],[.01,.99])
        self.assertFalse(r['accepted']);self.assertIsNone(r['class_id']);self.assertEqual(r['candidate_id'],'A')
        self.assertEqual(r['assist_status'],'cnn_disagrees')
    def test_cnn_alone_cannot_produce_output(self):
        with self.assertRaises(ValueError):select(None,[.99,.01])
    def test_uncertain_cnn_agreement_does_not_promote(self):
        self.assertFalse(select([.72,.28],[.6,.4])['accepted'])
    def test_very_weak_keypoint_is_not_rescued_by_cnn(self):
        self.assertFalse(select([.55,.45],[.99,.01])['accepted'])
    def test_missing_cnn_stays_rejected(self):
        r=select([.72,.28]);self.assertFalse(r['accepted']);self.assertTrue(r['assistance_needed'])
    def test_mode_check_skips_cnn_for_strong_keypoint(self):
        self.assertFalse(needs_mode_analysis(raw([.99,.01]),'assisted'))
        self.assertTrue(needs_mode_analysis(raw([.72,.28]),'assisted'))

class AssistedPipelineTests(unittest.TestCase):
    def model(self,kp,rgb=None,fail=False):
        model=SimpleNamespace(classes=CLASSES,min_score=.75,min_margin=.15,calls=[])
        def predict(rois,keypoint_frames=None,mode='keypoint'):
            model.calls.append(mode)
            if mode=='rgb':
                if fail:raise RuntimeError('CNN unavailable')
                data=raw(None,rgb)
                data['computed_modalities']=['rgb']
            else:data=raw(kp)
            return select_prediction(data,CLASSES,mode)
        model.predict_rois=predict
        return model
    def test_confident_fast_path_never_extracts_rois(self):
        model=self.model([.99,.01])
        with patch('fusion.fast.SharedExtractor') as heavy:r=analyze_clip(model,'.',clip_with_landmarks(),'assisted')
        heavy.assert_not_called();self.assertEqual(model.calls,['keypoint'])
        self.assertTrue(r['accepted']);self.assertEqual(r['pipeline'],'live_keypoints')
    def test_cnn_runs_only_for_uncertain_keypoint(self):
        model=self.model([.72,.28],[.99,.01])
        with patch('fusion.fast.SharedExtractor') as heavy:
            heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)}, {})
            r=analyze_clip(model,'.',clip_with_landmarks(),'assisted')
        heavy.return_value.extract.assert_called_once()
        self.assertEqual(model.calls,['keypoint','rgb']);self.assertTrue(r['accepted'])
    def test_offline_fallback_does_not_extract_twice(self):
        model=self.model([.72,.28],[.99,.01])
        with patch('fusion.fast.live_keypoints',side_effect=ValueError('No live tracking')),patch('fusion.fast.SharedExtractor') as heavy:
            heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)}, {})
            r=analyze_clip(model,'.',clip_with_landmarks(),'assisted')
        heavy.return_value.extract.assert_called_once();self.assertTrue(r['accepted'])
    def test_failed_auxiliary_cannot_force_output(self):
        model=self.model([.72,.28],fail=True)
        with patch('fusion.fast.SharedExtractor') as heavy:
            heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)}, {})
            r=analyze_clip(model,'.',clip_with_landmarks(),'assisted')
        self.assertFalse(r['accepted']);self.assertEqual(r['assist_status'],'cnn_unavailable')
        self.assertIn('rgb',r['component_errors']);self.assertFalse(needs_mode_analysis(r,'assisted'))
    def test_real_model_routing_does_not_construct_cnn_for_confident_keypoint(self):
        model=FusionRecognizer();p=np.zeros(30);p[0]=1
        with patch.object(model.keypoint,'predict_proba',return_value=p),patch('fusion.predict.RGBBranch',side_effect=AssertionError('CNN must remain unloaded')):
            r=model.predict_rois({},keypoint_frames=np.zeros((4,67,3)),mode='assisted')
        self.assertIsNone(model._rgb);self.assertTrue(r['accepted']);self.assertEqual(r['computed_modalities'],['keypoint'])

if __name__=='__main__':unittest.main()
