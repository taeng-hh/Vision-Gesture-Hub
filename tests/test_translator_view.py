"""Translator screen renders without a webcam in every state."""
import unittest
import numpy as np
from ui import theme
from ui.translator_view import (CAMERA_BOX, DEBUG_BOX, SUBTITLE_BOX, TranslatorState,
                                draw_translator)

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
