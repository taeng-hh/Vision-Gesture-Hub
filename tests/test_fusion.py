"""Fusion contracts: ID alignment, fallback, rejection and bounded recording."""
import unittest
import numpy as np
from fusion.predict import align_probabilities,fuse_probabilities,recognition_event,FusionRecognizer,select_prediction
from fusion.live import ClipRecorder
from fusion.shared import FrameClip

class FusionTests(unittest.TestCase):
    def test_class_order_is_aligned_by_id(self):
        actual=align_probabilities([.9,.09,.01],["C","B","A"],["A","B","C"])
        np.testing.assert_allclose(actual,[.01,.09,.9])
    def test_duplicate_or_missing_classes_fail(self):
        for a,b in [(["A","A"],["A","B"]),(["A","B"],["A","C"])]:
            with self.assertRaises(ValueError):align_probabilities([.5,.5],a,b)
    def test_bad_probabilities_fail(self):
        for a in [[float("nan"),.5],[-.1,1.1],[0.,0.]]:
            with self.assertRaises(ValueError):align_probabilities(a,["A","B"],["A","B"])
    def test_fusion_averages_probabilities_without_second_softmax(self):
        p,m=fuse_probabilities(np.array([.9,.1]),np.array([.2,.8]),.25)
        np.testing.assert_allclose(p,[.375,.625]);self.assertEqual(m,["keypoint","rgb"])
    def test_missing_component_uses_available_component(self):
        a=np.array([.2,.8]);p,m=fuse_probabilities(None,a,.9)
        np.testing.assert_array_equal(p,a);self.assertEqual(m,["rgb"])
        p,m=fuse_probabilities(a,None,.1)
        np.testing.assert_array_equal(p,a);self.assertEqual(m,["keypoint"])
        with self.assertRaises(ValueError):fuse_probabilities(None,None)
    def test_bad_weight_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):FusionRecognizer(keypoint_weight=1.1)
    def test_low_score_yields_no_recognition_event(self):
        classes=[dict(class_id="A",kind="word",text="a"),dict(class_id="B",kind="sentence",text="b")]
        r=recognition_event([.55,.45],classes)
        self.assertFalse(r["accepted"]);self.assertIsNone(r["class_id"]);self.assertEqual(r["candidate_id"],"A")
    def test_keypoint_mode_is_not_overruled_by_conflicting_rgb(self):
        classes=[dict(class_id="A",kind="word",text="a"),dict(class_id="B",kind="word",text="b")]
        raw=dict(keypoint_probabilities=[.95,.05],rgb_probabilities=[.01,.99],keypoint_weight=.1)
        kp=select_prediction(raw,classes,"keypoint")
        rgb=select_prediction(raw,classes,"rgb")
        combined=select_prediction(raw,classes,"fusion")
        self.assertEqual(kp["class_id"],"A");self.assertEqual(rgb["class_id"],"B");self.assertEqual(combined["class_id"],"B")
        self.assertEqual(kp["decision_keypoint_weight"],1.)
        self.assertNotIn("decision_mode",raw)
    def test_reselecting_fusion_uses_original_components(self):
        classes=[dict(class_id="A",kind="word",text="a"),dict(class_id="B",kind="word",text="b")]
        raw=dict(keypoint_probabilities=[.95,.05],rgb_probabilities=[.01,.99],keypoint_weight=.1)
        kp=select_prediction(raw,classes,"keypoint")
        actual=select_prediction(kp,classes,"fusion")
        np.testing.assert_allclose(actual["probabilities"],[.104,.896])
    def test_selected_missing_branch_is_not_silently_replaced(self):
        classes=[dict(class_id="A",kind="word",text="a"),dict(class_id="B",kind="word",text="b")]
        raw=dict(keypoint_probabilities=None,rgb_probabilities=[.01,.99],keypoint_weight=.1)
        with self.assertRaises(ValueError):select_prediction(raw,classes,"keypoint")
        self.assertEqual(select_prediction(raw,classes,"fusion")["class_id"],"B")
    def test_invalid_recognition_mode_is_rejected(self):
        with self.assertRaises(ValueError):select_prediction({},[],"unknown")
    def test_sentence_metadata_is_preserved(self):
        classes=[dict(class_id="A",kind="word",text="a"),dict(class_id="B",kind="sentence",text="문장")]
        r=recognition_event([.01,.99],classes)
        self.assertEqual((r["class_id"],r["kind"],r["text"]),("B","sentence","문장"))

class RecordingTests(unittest.TestCase):
    def test_idle_frames_not_recorded(self):
        r=ClipRecorder();r.update(np.zeros((4,4,3),np.uint8),1.)
        self.assertEqual(r.frames,[])
    def test_thirty_fps_jitter_does_not_halve_recording_rate(self):
        r=ClipRecorder();r.start(0);frame=np.zeros((4,4,3),np.uint8)
        for i in range(30):r.update(frame,i/30+(.0001 if i%2 else 0))
        self.assertEqual(len(r.frames),30)
    def test_frame_buffer_copies_camera_input(self):
        r=ClipRecorder();r.start(0);frame=np.zeros((4,4,3),np.uint8);r.update(frame,0);frame[:]=255
        self.assertFalse(r.frames[0].any())
    def test_memory_limit_cancels_without_partial_clip(self):
        r=ClipRecorder(max_bytes=50);r.start(0);frame=np.zeros((4,4,3),np.uint8);r.update(frame,0)
        with self.assertRaises(ValueError):r.update(frame,.2)
        self.assertFalse(r.recording);self.assertFalse(r.frames)
    def test_camera_gap_cancels(self):
        r=ClipRecorder();r.start(0);r.update(np.zeros((4,4,3),np.uint8),0)
        with self.assertRaises(ValueError):r.update(np.zeros((4,4,3),np.uint8),2)
        self.assertFalse(r.recording)
    def test_valid_clip_releases_recording_buffer(self):
        r=ClipRecorder();r.start(0)
        for i in range(4):r.update(np.full((4,4,3),i,np.uint8),i*.1)
        clip=r.finish();self.assertFalse(r.recording);self.assertEqual(len(clip.frames),4)
        self.assertEqual(r.frames,[])
    def test_duplicate_timestamps_rejected(self):
        with self.assertRaises(ValueError):FrameClip([np.zeros((4,4,3),np.uint8)]*4,[0,0,1,2])
    def test_timestamp_sampling_preserves_first_and_last_frame(self):
        frames=[np.full((4,4,3),i,np.uint8) for i in range(4)]
        cap=FrameClip(frames,[0,.1,.9,1.]).capture();values=[]
        while True:
            ok,f=cap.read()
            if not ok:break
            values.append(int(f[0,0,0]))
        self.assertEqual(values[0],0);self.assertEqual(values[-1],3)

if __name__=="__main__":unittest.main()
