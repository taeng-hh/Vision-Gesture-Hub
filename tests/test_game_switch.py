"""V hand shape -> Galaga -> return, on the default (assisted) sign screen."""
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from core.types import Hand, Observation
from fusion.game_switch import GameSwitch

V = Observation((Hand((), 'Victory', .95),))
EMPTY = Observation()
RAW = np.zeros((72, 128, 3), np.uint8)


class FakeDetector:
    def __init__(self, observation=V):
        self.observation, self.calls, self.closed = observation, 0, False
    def detect(self, frame, timestamp):
        self.calls += 1
        return self.observation
    def close(self):
        self.closed = True


def hold(switch, start=0., recording=False, seconds=1.6):
    return [switch.update(RAW, start + i * .1, recording) for i in range(int(seconds / .1))]


class GameSwitchTests(unittest.TestCase):
    def switch(self, detector=None, run_game=None):
        return GameSwitch(0, 'models/ksl30_robust_v2', 'models',
                          detector=detector or FakeDetector(), run_game=run_game)

    def test_hold_launches_once_until_release(self):
        switch = self.switch()
        self.assertEqual(sum(hold(switch)), 1)
        self.assertFalse(any(hold(switch, 2.)))

    def test_recording_blocks_and_requires_release(self):
        detector = FakeDetector()
        switch = self.switch(detector)
        self.assertFalse(any(hold(switch, recording=True)))
        self.assertFalse(any(hold(switch, 2.)))
        detector.observation = EMPTY
        switch.update(RAW, 4., False)
        detector.observation = V
        self.assertEqual(sum(hold(switch, 4.1)), 1)

    def test_detection_is_throttled(self):
        detector = FakeDetector(EMPTY)
        switch = self.switch(detector)
        for i in range(30):
            switch.update(RAW, i / 30, False)
        self.assertLessEqual(detector.calls, 13)

    def test_launch_passes_models_and_reports_failure(self):
        calls = []
        switch = self.switch(run_game=lambda *a: calls.append(a))
        self.assertIn('돌아왔습니다', switch.launch())
        camera, model, pose = calls[0]
        self.assertEqual((camera, model.name, pose.name), (0, 'ksl30_robust_v2', 'pose_landmarker_lite.task'))

        def broken(*_):
            raise FileNotFoundError('Required models')
        self.assertIn('실행 실패', self.switch(run_game=broken).launch())

    def test_missing_gesture_model_disables_switch_without_crashing(self):
        with tempfile.TemporaryDirectory() as empty:
            switch = GameSwitch(0, 'models/ksl30_robust_v2', empty)
        self.assertFalse(switch.update(RAW, 0., False))
        self.assertIn('게임 전환 꺼짐', switch.lines(False)[0])

    def test_progress_line_hidden_while_recording(self):
        switch = self.switch()
        hold(switch, seconds=.6)
        self.assertTrue(any('미니게임 열기' in line for line in switch.lines(False)))
        self.assertFalse(any('미니게임 열기' in line for line in switch.lines(True)))


class RunCameraGameTests(unittest.TestCase):
    def test_camera_is_handed_to_game_and_reopened(self):
        from fusion.live import run_camera
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.open.return_value = True
        cap.read.return_value = (True, RAW)
        tracker = MagicMock()
        tracker.snapshot.return_value = None
        game = MagicMock()
        game.update.side_effect = [True, False, False]
        game.launch.return_value = 'back'
        game.error = None
        game.progress = .5
        model = SimpleNamespace(classes=[dict(class_id='A', text='a')])
        views = []
        with ExitStack() as st:
            st.enter_context(patch('fusion.live.cv2.VideoCapture', return_value=cap))
            st.enter_context(patch('fusion.live.PreviewTracker', return_value=tracker))
            st.enter_context(patch('fusion.live.time.monotonic', return_value=1))
            for name in ['namedWindow', 'resizeWindow', 'destroyWindow', 'destroyAllWindows', 'imshow']:
                st.enter_context(patch('fusion.live.cv2.' + name))
            st.enter_context(patch('fusion.live.cv2.getWindowProperty', return_value=1))
            st.enter_context(patch('fusion.live.cv2.waitKey', side_effect=[-1, ord('q')]))
            st.enter_context(patch('fusion.live.draw_translator',
                                   side_effect=lambda state: views.append(state) or state.camera))
            run_camera(model, '.', camera=3, game=game)
        game.launch.assert_called_once_with(window='Sign Translator')  # Same window as the translator.
        cap.open.assert_called_once_with(3)
        self.assertEqual(cap.release.call_count, 2)  # Before the game and on exit.
        self.assertEqual(views[0].game_progress, .5)
        self.assertEqual(views[0].message, 'back')
        self.assertFalse(game.update.call_args_list[0].kwargs['recording'])


if __name__ == '__main__':
    unittest.main()
