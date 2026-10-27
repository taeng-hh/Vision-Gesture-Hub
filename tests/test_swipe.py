import unittest

from core.swipe import OpenPalmSwipe
from core.types import Hand, Landmark


def palm(x, y=0.5, gesture="Open_Palm", score=0.95):
    return Hand(tuple(Landmark(x, y, 0) for _ in range(21)),
                gesture, score, "Right")


class TestSwipe(unittest.TestCase):
    def test_shorter_swipe_from_near_center(self):
        gate = OpenPalmSwipe()
        self.assertFalse(gate.update(palm(0.60), 0))
        self.assertTrue(gate.update(palm(0.36), 0.10))

    def test_brief_label_dropout_keeps_swipe(self):
        gate = OpenPalmSwipe()
        gate.update(palm(0.7), 0)
        self.assertFalse(gate.update(palm(0.6, gesture="None", score=0), 0.05))
        self.assertTrue(gate.update(palm(0.45), 0.1))

    def swipe(self, gate, start=0):
        results = [gate.update(palm(x), start + i * 0.1)
                   for i, x in enumerate((0.8, 0.7, 0.6, 0.45, 0.35))]
        return sum(results)

    def test_leftward_swipe_triggers_once(self):
        gate = OpenPalmSwipe()
        self.assertEqual(self.swipe(gate), 1)
        self.assertEqual(self.swipe(gate, 1.5), 0)
        gate.update(None, 2)
        gate.update(None, 2.3)
        self.assertEqual(self.swipe(gate, 2.4), 1)

    def test_rightward_motion_does_not_trigger(self):
        gate = OpenPalmSwipe()
        for i, x in enumerate((0.2, 0.35, 0.5, 0.7, 0.8)):
            self.assertFalse(gate.update(palm(x), i * 0.1))

    def test_requires_open_palm_and_confidence(self):
        for gesture, score in (("Closed_Fist", 0.95), ("Open_Palm", 0.5)):
            gate = OpenPalmSwipe()
            for i, x in enumerate((0.8, 0.6, 0.4)):
                self.assertFalse(gate.update(palm(x, gesture=gesture, score=score), i * 0.1))

    def test_rejects_camera_gap_vertical_motion_and_slow_motion(self):
        for end_y, end_time in ((0.5, 1), (0.8, 0.2)):
            gate = OpenPalmSwipe()
            gate.update(palm(0.8), 0)
            self.assertFalse(gate.update(palm(0.4, end_y), end_time))
        gate = OpenPalmSwipe()
        for i in range(11):
            self.assertFalse(gate.update(palm(0.8 - i * 0.03), i * 0.2))
