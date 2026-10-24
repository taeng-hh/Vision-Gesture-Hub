import unittest
from core.temporal import OneShotGestureGate


class TestOneShotGate(unittest.TestCase):
    def test_one_shot_with_rearm(self):
        gate = OneShotGestureGate(stable_frames=3, neutral_frames=2, cooldown_s=0)
        allowed = {"Thumb_Up"}
        self.assertIsNone(gate.update("Thumb_Up", 0, allowed))
        self.assertIsNone(gate.update("Thumb_Up", 0.1, allowed))
        self.assertEqual(gate.update("Thumb_Up", 0.2, allowed), "Thumb_Up")
        for i in range(5):
            self.assertIsNone(gate.update("Thumb_Up", 0.3 + i*.1, allowed))
        gate.update("None", 1.0, allowed)
        gate.update("None", 1.1, allowed)
        gate.update("Thumb_Up", 2.0, allowed)
        gate.update("Thumb_Up", 2.1, allowed)
        self.assertEqual(gate.update("Thumb_Up", 2.2, allowed), "Thumb_Up")

    def test_rejects_unstable(self):
        gate = OneShotGestureGate(stable_frames=3)
        for i, g in enumerate(("Thumb_Up", "None", "Thumb_Up", "None")):
            self.assertIsNone(gate.update(g, i, {"Thumb_Up"}))
