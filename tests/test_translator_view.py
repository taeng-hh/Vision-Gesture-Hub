"""Translator screen renders without a webcam in every state."""
import unittest
import numpy as np
from ui import theme
from ui.translator_view import (CAMERA_BOX, CAMERA_VIEW, DEBUG_BOX, RESULT_BOX, SUBTITLE_BOX,
                                TranslatorState, draw_translator)

CAMERA = np.full((720, 1280, 3), 40, np.uint8)
WORD = dict(accepted=True, kind='word', text='행복', score=.91, candidates=[('행복', .91)])
RETRY = dict(accepted=False, kind=None, text='감사', score=.58, candidates=[('감사', .58), ('죄송', .31)])


def render(**kwargs):
    return draw_translator(TranslatorState(camera=kwargs.pop('camera', CAMERA), **kwargs))


def region(image, box):
    left, top, right, bottom = box
    return image[top:bottom, left:right]


class TranslatorViewTests(unittest.TestCase):
    def test_every_state_is_full_size_bgr(self):
        states = [dict(), dict(status='recording', elapsed=1.2, now=.2), dict(status='recording', now=.7),
                  dict(status='analyzing'), dict(event=WORD, tokens=('나', '행복'), sentence='저는 행복해요.'),
                  dict(event=RETRY), dict(debug=True, debug_lines=('a', 'b')), dict(game_progress=None, game_note='꺼짐'),
                  dict(camera=np.zeros((0, 0, 3), np.uint8)), dict(camera=np.zeros((480, 640, 3), np.uint8))]
        for state in states:
            image = render(**state)
            self.assertEqual(image.shape, (theme.HEIGHT, theme.WIDTH, 3), state)
            self.assertEqual(image.dtype, np.uint8)
            self.assertTrue(image.flags['C_CONTIGUOUS'])

    def test_background_uses_theme(self):
        image = render()
        self.assertEqual(tuple(image[5, 5]), theme.bgr(theme.BACKGROUND))

    def test_camera_pixels_reach_the_camera_card(self):
        bright = np.full((720, 1280, 3), 200, np.uint8)
        self.assertGreater(region(render(camera=bright), CAMERA_BOX).mean(), region(render(), CAMERA_BOX).mean())

    def test_recording_badge_is_red_and_blinks(self):
        recording = theme.bgr(theme.RECORDING)
        on = render(status='recording', elapsed=1., now=0.)
        off = render(status='recording', elapsed=1., now=.6)
        header = lambda image: image[28:88, 900:1260].reshape(-1, 3)
        has = lambda image, color: bool((header(image) == color).all(axis=1).any())
        self.assertTrue(has(on, recording))
        self.assertFalse((on == off).all())        # Dot toggles.
        self.assertFalse(has(render(), recording))  # Ready badge is mint.

    def test_tokens_and_sentence_change_the_subtitle(self):
        empty = region(render(), SUBTITLE_BOX)
        filled = region(render(tokens=('나', '행복'), sentence='저는 행복해요.'), SUBTITLE_BOX)
        self.assertFalse((empty == filled).all())

    def test_many_tokens_do_not_overflow(self):
        image = render(tokens=tuple(['배고프다']*30), sentence='가'*200)
        self.assertEqual(image.shape, (theme.HEIGHT, theme.WIDTH, 3))

    def test_debug_panel_is_hidden_by_default_and_toggles(self):
        rois = np.full((3, 224, 224, 3), 255, np.uint8)
        hidden = region(render(rois=rois, debug_lines=('키포인트: 행복 0.71',)), DEBUG_BOX)
        shown = region(render(rois=rois, debug_lines=('키포인트: 행복 0.71',), debug=True), DEBUG_BOX)
        self.assertFalse((hidden == shown).all())
        self.assertTrue((shown == 255).all(axis=2).any())  # ROI thumbnails only when open.
        self.assertFalse((hidden == 255).all(axis=2).any())

    def test_camera_frame_is_not_modified(self):
        frame = CAMERA.copy()
        render(camera=frame, event=WORD)
        np.testing.assert_array_equal(frame, CAMERA)


class DemoPolishTests(unittest.TestCase):
    def has(self, image, box, color):
        return bool((region(image, box).reshape(-1, 3) == theme.bgr(color)).all(axis=1).any())

    def test_new_result_flashes_then_settles(self):
        fresh = render(event=WORD, event_age=0., tokens=('행복',), sentence='행복')
        settled = render(event=WORD, event_age=5., tokens=('행복',), sentence='행복')
        border = (RESULT_BOX[0], RESULT_BOX[1], RESULT_BOX[2], RESULT_BOX[1]+3)
        self.assertTrue(self.has(fresh, border, theme.ACCENT))
        self.assertFalse(self.has(settled, border, theme.ACCENT))
        # Newest chip is filled mint only while fresh.
        self.assertGreater((region(fresh, SUBTITLE_BOX) == theme.bgr(theme.ACCENT)).all(axis=2).sum(),
                           (region(settled, SUBTITLE_BOX) == theme.bgr(theme.ACCENT)).all(axis=2).sum())

    def test_tracking_banner_appears_over_camera_only_when_needed(self):
        lower = (CAMERA_VIEW[0], CAMERA_VIEW[3]-70, CAMERA_VIEW[2], CAMERA_VIEW[3])
        ok = render(keypoints=True, tracking_note='')
        missing = render(keypoints=False, tracking_note='상반신을 보여주세요')
        self.assertFalse(self.has(ok, lower, theme.WARNING))
        self.assertTrue(self.has(missing, lower, theme.WARNING))
        self.assertTrue(self.has(render(keypoints=True, camera_note='카메라 다시 연결 중…'), lower, theme.WARNING))

    def test_recording_progress_bar_grows(self):
        bar = (CAMERA_VIEW[0], 604, CAMERA_VIEW[2], 613)
        count = lambda image: int((region(image, bar) == theme.bgr(theme.RECORDING)).all(axis=2).sum())
        early, late = render(status='recording', elapsed=1.), render(status='recording', elapsed=9.)
        self.assertGreater(count(late), count(early))
        self.assertEqual(count(render()), 0)

    def test_history_recent_overlay_and_reconnect_states_render(self):
        base = region(render(sentence='저는 행복해요.'), SUBTITLE_BOX)
        with_history = region(render(sentence='저는 행복해요.', history=('밥을 주세요.',)), SUBTITLE_BOX)
        self.assertFalse((base == with_history).all())
        self.assertFalse((region(render(), RESULT_BOX) == region(render(recent=(('감사', .9),)), RESULT_BOX)).all())
        for state in (dict(overlay=('SIGN GALAGA로 이동합니다', 'Q로 복귀')), dict(status='reconnecting', camera=None),
                      dict(status='analyzing', now=.4), dict(sentence='가'*120, tokens=('나',))):
            self.assertEqual(render(**state).shape, (theme.HEIGHT, theme.WIDTH, 3))


class CameraDropTests(unittest.TestCase):
    def run_loop(self, reads, keys, times):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from fusion.live import run_camera
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.open.return_value = True
        cap.read.side_effect = reads
        tracker = MagicMock()
        tracker.snapshot.return_value = None
        states = []
        with ExitStack() as st:
            st.enter_context(patch('fusion.live.cv2.VideoCapture', return_value=cap))
            st.enter_context(patch('fusion.live.PreviewTracker', return_value=tracker))
            st.enter_context(patch('fusion.live.time.monotonic', side_effect=times))
            for name in ['namedWindow', 'resizeWindow', 'destroyAllWindows', 'imshow']:
                st.enter_context(patch('fusion.live.cv2.' + name))
            st.enter_context(patch('fusion.live.cv2.getWindowProperty', return_value=1))
            st.enter_context(patch('fusion.live.cv2.waitKey', side_effect=keys))
            st.enter_context(patch('fusion.live.draw_translator',
                                   side_effect=lambda state: states.append(state) or np.zeros((1, 1, 3), np.uint8)))
            run_camera(SimpleNamespace(classes=[dict(class_id='A', text='a')]), '.')
        return cap, states

    def test_brief_drop_shows_reconnecting_and_recovers(self):
        good = (True, np.zeros((72, 128, 3), np.uint8))
        cap, states = self.run_loop([good, (False, None), (False, None), good],
                                    [-1, -1, -1, ord('q')], [0., .1, .2, .3])
        self.assertEqual([s.status for s in states], ['ready', 'reconnecting', 'reconnecting', 'ready'])
        self.assertIn('연결', states[1].camera_note)
        cap.release.assert_called_once()  # Only on exit; no reopen for a short drop.

    def test_long_drop_reopens_once_then_gives_up(self):
        from fusion.live import CAMERA_GRACE_SECONDS
        bad = (False, None)
        times = [0., .5, 1.2, 2., CAMERA_GRACE_SECONDS+.1]
        with self.assertRaisesRegex(RuntimeError, '카메라 입력이 중단'):
            self.run_loop([bad]*5, [-1]*5, times)

    def test_drop_cancels_recording(self):
        good = (True, np.zeros((72, 128, 3), np.uint8))
        _, states = self.run_loop([good, good, (False, None), good], [ord(' '), -1, -1, ord('q')], [0., .1, .2, .3])
        self.assertEqual(states[1].status, 'recording')
        self.assertEqual(states[2].status, 'reconnecting')
        self.assertIn('녹화를 취소', states[2].message)
        self.assertEqual(states[3].status, 'ready')


class LoopKeyTests(unittest.TestCase):
    def test_d_toggles_debug_panel_in_the_camera_loop(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from fusion.live import run_camera
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((72, 128, 3), np.uint8))
        tracker = MagicMock()
        tracker.snapshot.return_value = None
        states = []
        with ExitStack() as st:
            st.enter_context(patch('fusion.live.cv2.VideoCapture', return_value=cap))
            st.enter_context(patch('fusion.live.PreviewTracker', return_value=tracker))
            st.enter_context(patch('fusion.live.time.monotonic', return_value=1))
            for name in ['namedWindow', 'resizeWindow', 'destroyAllWindows', 'imshow']:
                st.enter_context(patch('fusion.live.cv2.' + name))
            st.enter_context(patch('fusion.live.cv2.getWindowProperty', return_value=1))
            st.enter_context(patch('fusion.live.cv2.waitKey', side_effect=[ord('d'), -1, ord('D'), ord('q')]))
            st.enter_context(patch('fusion.live.draw_translator',
                                   side_effect=lambda state: states.append(state) or state.camera))
            run_camera(SimpleNamespace(classes=[dict(class_id='A', text='a')]), '.')
        self.assertEqual([s.debug for s in states], [False, True, True, False])
        self.assertEqual(states[0].status, 'ready')


if __name__ == '__main__':
    unittest.main()
