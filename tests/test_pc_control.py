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
    def test_paint_launch_and_gesture_change_restarts_hold(self):
        launcher = Mock(return_value=True)
        pc = PCController(backend=self.mouse, launcher=launcher)
        victory = Hand(hand(0.8).landmarks, "Victory", 0.95, "Right")
        paint = Hand(hand(0.8).landmarks, "ILoveYou", 0.95, "Right")
        pc.update(victory, 0)
        pc.update(paint, 0.8)
        pc.update(paint, 1.1)
        launcher.assert_not_called()
        pc.update(paint, 1.9)
        pc.update(paint, 8)
        launcher.assert_called_once_with("paint")
        self.mouse.moveTo.assert_not_called()
        self.mouse.click.assert_not_called()

    def test_chatgpt_requires_hold_and_release_without_mouse_input(self):
        launcher = Mock(return_value=True)
        pc = PCController(backend=self.mouse, launcher=launcher)
        victory = Hand(hand(0.8).landmarks, "Victory", 0.95, "Right")
        pc.update(victory, 0)
        pc.update(victory, 0.9)
        launcher.assert_not_called()
        pc.update(victory, 1.1)
        pc.update(victory, 8)
        launcher.assert_called_once_with("chatgpt")
        self.mouse.moveTo.assert_not_called()
        self.mouse.click.assert_not_called()
        pc.update(None, 8.1)
        pc.update(victory, 8.2)  # Brief tracking loss must not rearm.
        pc.update(victory, 9.3)
        self.assertEqual(launcher.call_count, 1)
        pc.update(None, 9.4)
        pc.update(None, 9.8)
        pc.update(victory, 10)
        pc.update(victory, 11.1)
        self.assertEqual(launcher.call_count, 2)

    def test_chatgpt_ignores_low_confidence_and_handles_browser_failure(self):
        launcher = Mock(return_value=False)
        pc = PCController(backend=self.mouse, launcher=launcher)
        weak = Hand(hand(0.8).landmarks, "Victory", 0.6, "Right")
        pc.update(weak, 0)
        pc.update(weak, 2)
        launcher.assert_not_called()
        victory = Hand(hand(0.8).landmarks, "Victory", 0.95, "Right")
        pc.update(victory, 3)
        self.assertIn("Could not open", pc.update(victory, 4.1))

    def test_open_palm_swipe_switches_app_without_mouse_input(self):
        for i, x in enumerate((0.8, 0.7, 0.6, 0.45, 0.35)):
            palm = Hand(tuple(Landmark(x, 0.5, 0) for _ in range(21)),
                        "Open_Palm", 0.95, "Right")
            self.pc.update(palm, i * 0.1)
        self.mouse.hotkey.assert_called_once_with("alt", "tab")
        self.mouse.moveTo.assert_not_called()
        self.mouse.click.assert_not_called()

    def setUp(self):
        self.mouse = Mock()
        self.mouse.size.return_value = (1920, 1080)
        self.pc = PCController(backend=self.mouse)

    def arm(self, start=0):
        self.pc.update(hand(0.8), start)
        self.pc.update(hand(0.8), start + 0.12)

    def test_initial_pinch_never_clicks(self):
        self.pc.update(hand(0.2), 0)
        self.pc.update(hand(0.2), 1)
        self.mouse.click.assert_not_called()

    def test_click_once_and_freeze_then_rearm(self):
        self.arm()
        self.mouse.moveTo.reset_mock()
        self.pc.update(hand(0.3, 0.7), 0.2)
        self.pc.update(hand(0.3, 0.7), 0.3)
        self.pc.update(hand(0.3, 0.7), 1)
        self.mouse.click.assert_called_once()
        self.mouse.moveTo.assert_not_called()
        self.arm(1.1)
        self.pc.update(hand(0.3), 1.3)
        self.pc.update(hand(0.3), 1.4)
        self.assertEqual(self.mouse.click.call_count, 2)

    def test_short_pinch_is_ignored(self):
        self.arm()
        self.pc.update(hand(0.3), 0.2)
        self.pc.update(hand(0.8), 0.23)
        self.mouse.click.assert_not_called()

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
        self.mouse.click.assert_not_called()

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
        self.mouse.click.assert_not_called()

    def test_middle_pinch_clicks_with_index_extended(self):
        self.arm()
        points = list(hand(0.3).landmarks)
        points[8] = Landmark(0.5, 0.1, 0)
        self.pc.update(Hand(tuple(points)), 0.2)
        self.pc.update(Hand(tuple(points)), 0.3)
        self.mouse.click.assert_called_once()
