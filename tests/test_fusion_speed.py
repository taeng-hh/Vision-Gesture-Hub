"""Speed paths must preserve input boundaries and skip unused heavy branches."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import unittest
import numpy as np
from fusion.shared import FrameClip
from fusion.fast import live_keypoints,analyze_clip,needs_mode_analysis
from fusion.live import ClipRecorder
from fusion.predict import FusionRecognizer


def clip_with_landmarks():
    clip=FrameClip([np.zeros((24,24,3),np.uint8) for _ in range(31)],np.linspace(0,1,31))
    a=np.zeros((67,3),np.float32);a[44]=[300,200,1];a[47]=[500,200,1]
    for j in range(21):a[j]=[300+j*3,300+(j%5)*4,1]
    clip.landmark_samples=tuple((float(t),a.copy()) for t in clip.times)
    return clip

class SpeedTests(unittest.TestCase):
    def test_live_path_skips_media_pipe_and_rgb_preprocessing(self):
        clip=clip_with_landmarks();model=MagicMock();model.predict_rois.return_value={}
        with patch('fusion.fast.SharedExtractor') as heavy:
            result=analyze_clip(model,'.',clip,'keypoint')
        heavy.assert_not_called();self.assertEqual(result['pipeline'],'live_keypoints')
        self.assertEqual(model.predict_rois.call_args.kwargs['mode'],'keypoint')
        self.assertEqual(model.predict_rois.call_args.kwargs['keypoint_frames'].shape,(64,67,3))
    def test_tracking_gaps_use_original_offline_path(self):
        clip=clip_with_landmarks();clip.landmark_samples=clip.landmark_samples[::10]
        with self.assertRaises(ValueError):live_keypoints(clip)
        model=MagicMock();model.predict_rois.return_value={}
        with patch('fusion.fast.SharedExtractor') as heavy:
            heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)}, {})
            result=analyze_clip(model,'.',clip,'keypoint')
        heavy.return_value.extract.assert_called_once_with(clip)
        self.assertEqual(result['pipeline'],'offline_keypoint_fallback')
        self.assertEqual(model.predict_rois.call_args.kwargs['mode'],'keypoint')
    def test_reuse_can_be_disabled_for_accuracy_comparison(self):
        clip=clip_with_landmarks();model=MagicMock();model.predict_rois.return_value={}
        with patch('fusion.fast.SharedExtractor') as heavy:
            heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)}, {})
            result=analyze_clip(model,'.',clip,'keypoint',reuse_live=False)
        heavy.return_value.extract.assert_called_once_with(clip)
        self.assertEqual(result['pipeline'],'offline_keypoint_fallback')
        self.assertIsNone(model.predict_rois.call_args.kwargs['keypoint_frames'])
    def test_idle_landmarks_do_not_leak_into_next_recording(self):
        recorder=ClipRecorder();recorder.start(1)
        points=clip_with_landmarks().landmark_samples[0][1]
        recorder.update(np.zeros((24,24,3),np.uint8),1.1,SimpleNamespace(captured_at=.9,keypoints=points))
        self.assertEqual(len(recorder.landmark_samples),0)
        recorder.update(np.zeros((24,24,3),np.uint8),1.2,SimpleNamespace(captured_at=1.15,keypoints=points))
        self.assertEqual(list(recorder.landmark_samples),[1.15])
        recorder.cancel();self.assertEqual(recorder.landmark_samples,{})
    def test_missing_shoulders_fall_back_instead_of_inventing_points(self):
        clip=clip_with_landmarks()
        for _,a in clip.landmark_samples:a[44,2]=0
        with self.assertRaises(ValueError):live_keypoints(clip)
    def test_comparison_loads_missing_rgb_even_if_fusion_could_fallback(self):
        result=dict(computed_modalities=['keypoint'],available_modalities=['keypoint'])
        self.assertFalse(needs_mode_analysis(result,'keypoint'))
        self.assertTrue(needs_mode_analysis(result,'rgb'));self.assertTrue(needs_mode_analysis(result,'fusion'))
        result['computed_modalities'].append('rgb');self.assertFalse(needs_mode_analysis(result,'fusion'))
    def test_keypoint_mode_never_constructs_rgb_model(self):
        frames=live_keypoints(clip_with_landmarks())
        with patch('fusion.predict.RGBBranch',side_effect=AssertionError('RGB must stay unloaded')):
            model=FusionRecognizer();result=model.predict_rois({},keypoint_frames=frames,mode='keypoint')
        self.assertIsNone(model._rgb);self.assertEqual(result['computed_modalities'],['keypoint'])
    def test_rgb_mode_does_not_run_keypoint_classifier(self):
        model=FusionRecognizer();rgb=MagicMock();rgb.ids=model.ids;rgb.predict_proba.return_value=np.ones(30)/30;model._rgb=rgb
        with patch.object(model.keypoint,'predict_proba',side_effect=AssertionError('Unused keypoint classifier')):
            result=model.predict_rois({},mode='rgb')
        self.assertEqual(result['computed_modalities'],['rgb']);rgb.predict_proba.assert_called_once()

class LiveSpeedFlowTests(unittest.TestCase):
    def test_fast_then_rgb_comparison_runs_missing_branch_without_duplicate_output(self):
        from contextlib import ExitStack
        from concurrent.futures import Future
        from fusion.live import run_camera
        from fusion.predict import select_prediction
        class ImmediateExecutor:
            def __init__(self,**kwargs):pass
            def submit(self,fn,*args):
                f=Future()
                try:f.set_result(fn(*args))
                except Exception as exc:f.set_exception(exc)
                return f
            def shutdown(self,**kwargs):pass
        classes=[dict(class_id='A',text='a',kind='word'),dict(class_id='B',text='b',kind='word')]
        model=SimpleNamespace(classes=classes,ids=['A','B'],min_score=.75,min_margin=.15)
        modes=[]
        def predict(rois,keypoint_frames=None,mode='fusion'):
            modes.append(mode)
            raw=dict(keypoint_probabilities=[.99,.01] if mode!='rgb' else None,rgb_probabilities=[.01,.99] if mode!='keypoint' else None,keypoint_weight=.1,computed_modalities=['keypoint'] if mode=='keypoint' else ['keypoint','rgb'],available_modalities=['keypoint'],component_errors={})
            return select_prediction(raw,classes,mode)
        model.predict_rois=predict
        index=[-1];raw=np.zeros((24,24,3),np.uint8);points=clip_with_landmarks().landmark_samples[0][1]
        cap=MagicMock();cap.isOpened.return_value=True
        def read():index[0]+=1;return True,raw
        cap.read.side_effect=read
        tracker=MagicMock();tracker.snapshot.side_effect=lambda:SimpleNamespace(captured_at=index[0]*.05,keypoints=points)
        keys=[32,-1,-1,-1,-1,-1,32,-1,ord('2'),-1,ord('q')]
        quality=dict(left_direct_detection_fraction=1.,right_direct_detection_fraction=0.)
        with ExitStack() as st:
            st.enter_context(patch('fusion.live.cv2.VideoCapture',return_value=cap));st.enter_context(patch('fusion.live.PreviewTracker',return_value=tracker));st.enter_context(patch('fusion.live.ThreadPoolExecutor',ImmediateExecutor));st.enter_context(patch('fusion.live.time.monotonic',side_effect=lambda:index[0]*.05))
            for name in ['namedWindow','resizeWindow','destroyAllWindows','draw_skeleton']:st.enter_context(patch('fusion.live.'+('cv2.' if name!='draw_skeleton' else '')+name))
            st.enter_context(patch('fusion.live.tracking_state',return_value=dict(hands=1,shoulders=True,keypoints=True,tracking_note='')));st.enter_context(patch('fusion.live.cv2.getWindowProperty',return_value=1));st.enter_context(patch('fusion.live.cv2.waitKey',side_effect=keys));st.enter_context(patch('fusion.live.cv2.imshow'))
            views=[];st.enter_context(patch('fusion.live.draw_translator',side_effect=lambda state:(views.append(state) or state.camera)))
            heavy=st.enter_context(patch('fusion.fast.SharedExtractor'));heavy.return_value.extract.return_value=({'images':np.zeros((1,3,224,224,3),np.uint8)},quality)
            run_camera(model,'.')
        self.assertEqual(modes,['keypoint','rgb']);heavy.return_value.extract.assert_called_once()
        self.assertEqual(views[-1].tokens,('a',));self.assertEqual(views[-1].sentence,'a')
        self.assertTrue(any('영상: b' in text for text in views[-1].debug_lines))

if __name__=='__main__':unittest.main()
