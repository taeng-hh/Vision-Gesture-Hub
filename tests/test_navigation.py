import unittest

from core.mode_manager import Mode, ModeManager
from core.navigation import GestureNavigation
from core.types import Hand, Landmark


def hand(x, y):
    return Hand(tuple(Landmark(x, y, 0) for _ in range(21)))


class TestNavigation(unittest.TestCase):
    def hold(self, nav, point, mode, start=0):
        event = None
        for i in range(15):
            event = nav.update(point, mode, start + i * 0.1) or event
        return event

    def test_each_card_selects_its_mode(self):
        for x, mode in ((0.15, Mode.SIGN), (0.5, Mode.PRESENTATION),
                        (0.85, Mode.PC)):
            nav = GestureNavigation()
            self.assertEqual(self.hold(nav, hand(x, 0.6), Mode.IDLE), mode.value)

    def test_missing_hand_cancels_dwell(self):
        nav = GestureNavigation()
        nav.update(hand(0.15, 0.6), Mode.IDLE, 0)
        nav.update(None, Mode.IDLE, 0.1)
        self.assertIsNone(nav.update(hand(0.15, 0.6), Mode.IDLE, 0.2))
        self.assertEqual(nav.progress, 0)

    def test_camera_gap_does_not_select(self):
        nav = GestureNavigation()
        nav.update(hand(0.15, 0.6), Mode.IDLE, 0)
        self.assertIsNone(nav.update(hand(0.15, 0.6), Mode.IDLE, 2))

    def test_toggle_requires_leaving_before_repeating(self):
        nav = GestureNavigation()
        point = hand(0.8, 0.08)
        self.assertEqual(self.hold(nav, point, Mode.PC), "TOGGLE")
        self.assertIsNone(self.hold(nav, point, Mode.PC, 1.5))
        nav.update(hand(0.5, 0.5), Mode.PC, 3)
        self.assertEqual(self.hold(nav, point, Mode.PC, 3.1), "TOGGLE")

    def test_home_disables_os_actions(self):
        manager = ModeManager()
        manager.select(Mode.PC)
        manager.toggle_actions()
        event = self.hold(GestureNavigation(), hand(0.1, 0.08), manager.mode)
        self.assertEqual(event, "HOME")
        manager.select(Mode.IDLE)
        self.assertFalse(manager.actions_enabled)

    def test_sign_mode_has_no_enable_button(self):
        self.assertIsNone(self.hold(GestureNavigation(), hand(0.8, 0.08), Mode.SIGN))
