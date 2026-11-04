"""Preview geometry and diagnostic states must not alter recorded pixels."""
import unittest
from types import SimpleNamespace
import numpy as np
from fusion.preview import TrackingSnapshot,draw_skeleton,tracking_status,snapshot_from_results,diagnostic_view
from fusion.live import ClipRecorder

class PreviewTests(unittest.TestCase):
    def test_mirrored_hand_is_drawn_without_touching_recording(self):
        raw=np.zeros((100,200,3),np.uint8);recorder=ClipRecorder();recorder.start(1);recorder.update(raw,1)
        hand=tuple((.2,.4) for _ in range(21));snapshot=TrackingSnapshot(1,(hand,))
        display=raw.copy();draw_skeleton(display,snapshot,1)
        self.assertGreater(int(display[40,159].sum()),0)
        self.assertEqual(int(display[40,40].sum()),0)
        self.assertFalse(raw.any());self.assertFalse(recorder.frames[0].any())
    def test_stale_tracking_does_not_draw_old_hand(self):
        image=np.zeros((100,200,3),np.uint8);snapshot=TrackingSnapshot(1,(tuple((.2,.4) for _ in range(21)),))
        draw_skeleton(image,snapshot,2)
        self.assertFalse(image.any());self.assertIn('현재 위치 미확인',tracking_status(snapshot,2)[0])
    def test_missing_shoulders_and_hands_are_visible_diagnostics(self):
        empty=SimpleNamespace(hand_landmarks=[]);pose=SimpleNamespace(pose_landmarks=[])
        snapshot=snapshot_from_results(empty,pose,1280,720,1)
        self.assertFalse(snapshot.shoulders_ok);self.assertFalse(snapshot.keypoint_ok)
        text=' '.join(tracking_status(snapshot,1))
        self.assertIn('손 검출: 0/2',text);self.assertIn('상반신',text)
    def test_error_is_reported_and_no_invalid_skeleton_is_drawn(self):
        snapshot=TrackingSnapshot(1,error='test detector failure');image=np.zeros((100,200,3),np.uint8)
        draw_skeleton(image,snapshot,1)
        self.assertFalse(image.any());self.assertIn('test detector failure',tracking_status(snapshot,1)[0])
    def test_side_panel_preserves_all_image_pixels(self):
        image=np.full((100,200,3),125,np.uint8);original=image.copy()
        view=diagnostic_view(image,['손 검출: 2/2','양어깨: 확인됨'])
        np.testing.assert_array_equal(view[:100,:200],original)
        np.testing.assert_array_equal(image,original)

    def test_roi_panel_shows_rgb_crops_in_bgr_without_modifying_them(self):
        image=np.full((506,900,3),125,np.uint8)
        rois=np.zeros((3,224,224,3),np.uint8);rois[0,:,:,0]=255
        original=rois.copy();view=diagnostic_view(image,['input crops'],rois)
        self.assertEqual(tuple(view[550,40]),(0,0,255))
        np.testing.assert_array_equal(rois,original)
        np.testing.assert_array_equal(view[:506,:900],image)
    def test_live_window_renders_skeleton_and_cleans_up_without_camera(self):
        from unittest.mock import MagicMock,patch
        from contextlib import ExitStack
        from fusion.live import run_camera
        raw=np.zeros((200,300,3),np.uint8)
        cap=MagicMock();cap.isOpened.return_value=True;cap.read.return_value=(True,raw)
        tracker=MagicMock();tracker.snapshot.return_value=TrackingSnapshot(1,(tuple((.2,.4) for _ in range(21)),))
        model=SimpleNamespace(classes=[dict(class_id="WORD1290",text="감사")])
        with ExitStack() as stack:
            stack.enter_context(patch('fusion.live.cv2.VideoCapture',return_value=cap))
            stack.enter_context(patch('fusion.live.PreviewTracker',return_value=tracker))
            stack.enter_context(patch('fusion.live.time.monotonic',return_value=1))
            for name in ['namedWindow','resizeWindow','destroyAllWindows']:stack.enter_context(patch('fusion.live.cv2.'+name))
            show=stack.enter_context(patch('fusion.live.cv2.imshow'))
            stack.enter_context(patch('fusion.live.cv2.waitKey',return_value=ord('q')))
            run_camera(model,'.')
        image=show.call_args.args[1]
        self.assertGreater(int(image[80,239].sum()),0)
        self.assertFalse(raw.any());cap.release.assert_called_once();tracker.close.assert_called_once()

if __name__=='__main__':unittest.main()
