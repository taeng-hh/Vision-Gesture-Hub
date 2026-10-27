import unittest
from unittest.mock import Mock

from core.types import Hand, Landmark
from modes.pc_control import PCController


def hand(ratio, x=0.5):
    points = [Landmark(x, 0.5, 0) for _ in range(21)]
    points[5] = Landmark(0.4, 0.5, 0)
    points[17] = Landmark(0.6, 0.5, 0)
    points[12] = Landmark(x, 0.5, 0)
    points[4] = Landmark(x + ratio * 0.2, 0.5, 0)
    return Hand(tuple(points))


class TestPCController(unittest.TestCase):
    def test_open_palm_swipe_switches_app_without_mouse_input(self):
        for i, x in enumerate((0.8, 0.7, 0.6, 0.45, 0.35)):
            palm = Hand(tuple(Landmark(x, 0.5, 0) for _ in range(21)),
                        "Open_Palm", 0.95, "Right")
            self.pc.update(palm, i * 0.1)
        self.mouse.keyDown.assert_called_once_with("alt")
        self.mouse.press.assert_called_once_with("tab")
        self.mouse.keyUp.assert_not_called()
        self.mouse.moveTo.assert_not_called()
        self.mouse.mouseDown.assert_not_called()

    def swipe(self, start=0):
        for i, x in enumerate((0.8, 0.7, 0.6, 0.45, 0.35)):
            palm = Hand(tuple(Landmark(x, 0.5, 0) for _ in range(21)),
                        "Open_Palm", 0.95, "Right")
            self.pc.update(palm, start + i * 0.1)

    def test_repeated_swipes_cycle_in_same_window_picker(self):
        self.swipe()
        self.pc.update(None, 0.5)
        self.mouse.keyUp.assert_not_called()
        self.pc.update(hand(0.3), 0.6)
        self.swipe(1.2)
        self.swipe(2.4)
        self.mouse.keyDown.assert_called_once_with("alt")
        self.assertEqual(self.mouse.press.call_count, 3)
        self.mouse.moveTo.assert_not_called()
        self.mouse.mouseDown.assert_not_called()
        self.mouse.keyUp.assert_not_called()
        self.pc.update(None, 5.3)
        self.mouse.keyUp.assert_called_once_with("alt")
        self.assertFalse(self.pc._alt_down)

    def test_reset_releases_alt_even_at_failsafe_corner(self):
        self.swipe()
        def release(key):
            self.assertEqual(key, "alt")
            self.assertFalse(self.mouse.FAILSAFE)
        self.mouse.keyUp.side_effect = release
        self.pc.reset()
        self.pc.reset()
        self.mouse.keyUp.assert_called_once_with("alt")
        self.assertTrue(self.mouse.FAILSAFE)

    def test_switch_input_failure_releases_alt(self):
        self.mouse.press.side_effect = RuntimeError("input failed")
        with self.assertRaises(RuntimeError):
            self.swipe()
        self.mouse.keyUp.assert_called_once_with("alt")
        self.assertFalse(self.pc._alt_down)

    def fist(self, now, score=0.95, handedness="Right"):
        return self.pc.update(Hand(hand(0.3).landmarks, "Closed_Fist",
                                   score, handedness), now)

    def test_held_fist_confirms_highlighted_window_without_click(self):
        self.swipe()
        for now in (0.5, 0.6, 0.7):
            self.fist(now)
        self.mouse.keyUp.assert_not_called()
        self.assertIn("fist confirmed", self.fist(0.81))
        self.mouse.keyUp.assert_called_once_with("alt")
        self.mouse.press.assert_called_once_with("tab")
        self.mouse.mouseDown.assert_not_called()
        self.mouse.moveTo.assert_not_called()
        # Continuing to hold a fist cannot click after confirmation.
        self.fist(0.9)
        self.fist(1.1)
        self.mouse.mouseDown.assert_not_called()

    def test_fist_hold_restarts_after_interruption(self):
        self.swipe()
        self.fist(0.5)
        self.fist(0.65)
        self.pc.update(None, 0.7)
        self.fist(0.8)
        self.fist(0.95)
        self.mouse.keyUp.assert_not_called()
        self.fist(1.11)
        self.mouse.keyUp.assert_called_once_with("alt")

    def test_uncertain_fist_tracking_gap_and_hand_change_do_not_confirm(self):
        self.swipe()
        for now in (0.5, 0.65, 0.8, 0.95):
            self.fist(now, score=0.5)
        self.fist(1.0)
        self.fist(1.4)
        self.fist(1.55, handedness="Left")
        self.fist(1.7, handedness="Left")
        self.mouse.keyUp.assert_not_called()

    def setUp(self):
        self.mouse = Mock()
        self.mouse.size.return_value = (1920, 1080)
        self.pc = PCController(backend=self.mouse)

    def thumb(self, now, gesture="Thumb_Up", score=0.95):
        return self.pc.update(Hand(hand(0.8).landmarks, gesture, score, "Right"), now)

    def test_zoom_requires_hold_and_limits_repeat_rate(self):
        for now in (0, 0.15, 0.3, 0.45):
            self.thumb(now)
        self.mouse.hotkey.assert_not_called()
        self.thumb(0.51)
        self.mouse.hotkey.assert_called_once_with("win", "add")
        for now in (0.65, 0.8, 0.95, 1.1):
            self.thumb(now)
        self.assertEqual(self.mouse.hotkey.call_count, 1)
        self.thumb(1.12)
        self.assertEqual(self.mouse.hotkey.call_count, 2)
        self.mouse.mouseDown.assert_not_called()
        self.mouse.moveTo.assert_not_called()

    def test_zoom_out_and_tracking_gap_restart_hold(self):
        self.thumb(0, "Thumb_Down")
        self.thumb(0.4, "Thumb_Down")
        self.mouse.hotkey.assert_not_called()
        for now in (0.55, 0.7, 0.85, 0.96):
            self.thumb(now, "Thumb_Down")
        self.mouse.hotkey.assert_called_once_with("win", "subtract")

    def test_zoom_releases_drag_and_requires_pointer_rearming(self):
        self.start_drag()
        self.thumb(0.4)
        self.mouse.mouseUp.assert_called_once_with(button="left")
        self.pc.update(hand(0.3), 0.5)
        self.pc.update(hand(0.3), 0.7)
        self.mouse.mouseDown.assert_called_once()
        self.mouse.hotkey.assert_not_called()

    def test_zoom_disabled_during_window_selection(self):
        self.swipe()
        for now in (0.5, 0.65, 0.8, 0.95, 1.1):
            self.thumb(now)
        self.mouse.hotkey.assert_not_called()

    def arm(self, start=0):
        self.pc.update(hand(0.8), start)
        self.pc.update(hand(0.8), start + 0.12)

    def test_initial_pinch_never_clicks(self):
        self.pc.update(hand(0.2), 0)
        self.pc.update(hand(0.2), 1)
        self.mouse.mouseDown.assert_not_called()

    def test_hold_moves_pointer_then_release_and_rearm(self):
        self.arm()
        self.mouse.moveTo.reset_mock()
        self.pc.update(hand(0.3, 0.7), 0.2)
        self.pc.update(hand(0.3, 0.7), 0.3)
        self.pc.update(hand(0.3, 0.8), 1)
        self.mouse.mouseDown.assert_called_once_with(button="left")
        self.assertGreater(self.mouse.moveTo.call_count, 0)
        self.mouse.mouseUp.assert_not_called()
        self.arm(1.1)
        self.mouse.mouseUp.assert_called_once_with(button="left")
        self.pc.update(hand(0.3), 1.3)
        self.pc.update(hand(0.3), 1.4)
        self.assertEqual(self.mouse.mouseDown.call_count, 2)

    def start_drag(self):
        self.arm()
        self.pc.update(hand(0.3), 0.2)
        self.pc.update(hand(0.3), 0.3)
        self.mouse.mouseDown.assert_called_once_with(button="left")

    def test_hand_loss_releases_button_and_requires_open_fingers(self):
        self.start_drag()
        self.pc.update(None, 0.4)
        self.mouse.mouseUp.assert_called_once_with(button="left")
        self.pc.update(hand(0.3), 0.5)
        self.pc.update(hand(0.3), 0.7)
        self.mouse.mouseDown.assert_called_once()

    def test_reset_releases_once_even_at_failsafe_corner(self):
        self.start_drag()
        def release(**kwargs):
            self.assertFalse(self.mouse.FAILSAFE)
        self.mouse.mouseUp.side_effect = release
        self.pc.reset()
        self.pc.reset()
        self.mouse.mouseUp.assert_called_once_with(button="left")
        self.assertTrue(self.mouse.FAILSAFE)

    def test_swipe_pose_releases_drag(self):
        self.start_drag()
        palm = Hand(tuple(Landmark(0.8, 0.5, 0) for _ in range(21)),
                    "Open_Palm", 0.95, "Right")
        self.pc.update(palm, 0.4)
        self.mouse.mouseUp.assert_called_once_with(button="left")

    def test_hysteresis_keeps_button_held_until_fingers_open(self):
        self.start_drag()
        self.pc.update(hand(0.5), 0.4)
        self.mouse.mouseUp.assert_not_called()
        self.pc.update(hand(0.8), 0.5)
        self.mouse.mouseUp.assert_called_once_with(button="left")

    def test_short_pinch_is_ignored(self):
        self.arm()
        self.pc.update(hand(0.3), 0.2)
        self.pc.update(hand(0.8), 0.23)
        self.mouse.mouseDown.assert_not_called()

    def test_click_stays_fixed_during_debounce_and_held_jitter(self):
        self.arm()
        self.mouse.moveTo.reset_mock()
        self.pc.update(hand(0.3, 0.6), 0.2)
        self.pc.update(hand(0.3, 0.602), 0.3)
        for i in range(10):
            self.pc.update(hand(0.3, 0.6 + (-1) ** i * 0.002), 0.4 + i * 0.03)
        self.mouse.moveTo.assert_not_called()
        self.mouse.mouseDown.assert_called_once_with(button="left")
        self.pc.update(hand(0.8, 0.6), 0.8)
        self.mouse.mouseUp.assert_called_once_with(button="left")

    def test_drag_starts_only_after_threshold_and_stays_active(self):
        self.start_drag()
        self.mouse.moveTo.reset_mock()
        self.pc.update(hand(0.3, 0.503), 0.4)
        self.mouse.moveTo.assert_not_called()
        self.pc.update(hand(0.3, 0.55), 0.5)
        self.mouse.moveTo.assert_called_once()
        self.assertTrue(self.pc._dragging)
        self.pc.update(hand(0.3, 0.5), 0.6)
        self.assertTrue(self.pc._dragging)
        self.pc.update(None, 0.7)
        self.assertFalse(self.pc._dragging)
        self.assertIsNone(self.pc._pinch_origin)

    def test_invalid_drag_threshold(self):
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PCController(backend=self.mouse, drag_threshold=threshold)

    def test_small_jitter_does_not_move_cursor(self):
        self.arm()
        self.mouse.moveTo.reset_mock()
        for i in range(1, 31):
            self.pc.update(hand(0.8, 0.5 + (-1) ** i * 0.0005), 0.12 + i / 30)
        self.mouse.moveTo.assert_not_called()

    def test_hand_loss_requires_rearming(self):
        self.arm()
        self.pc.update(None, 0.2)
        self.pc.update(hand(0.3), 0.3)
        self.pc.update(hand(0.3), 0.5)
        self.mouse.mouseDown.assert_not_called()

    def test_large_movement_still_moves(self):
        self.arm()
        self.mouse.moveTo.reset_mock()
        self.pc.update(hand(0.8, 0.8), 0.16)
        self.mouse.moveTo.assert_called_once()

    def test_thumb_index_contact_does_not_click(self):
        self.arm()
        points = list(hand(0.8).landmarks)
        points[8] = points[4]
        self.pc.update(Hand(tuple(points)), 0.2)
        self.pc.update(Hand(tuple(points)), 0.4)
        self.mouse.mouseDown.assert_not_called()

    def test_middle_pinch_clicks_with_index_extended(self):
        self.arm()
        points = list(hand(0.3).landmarks)
        points[8] = Landmark(0.5, 0.1, 0)
        self.pc.update(Hand(tuple(points)), 0.2)
        self.pc.update(Hand(tuple(points)), 0.3)
        self.mouse.mouseDown.assert_called_once_with(button="left")
