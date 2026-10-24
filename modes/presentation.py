"""Control the currently active presentation window with two safe gestures."""
import pyautogui
from core.temporal import OneShotGestureGate


class PresentationController:
    ALLOWED = {"Thumb_Up", "Thumb_Down"}

    def __init__(self):
        self.gate = OneShotGestureGate(stable_frames=8, neutral_frames=3, cooldown_s=1.2)
        pyautogui.FAILSAFE = True

    def reset(self):
        self.gate.reset()

    def update(self, hand, now_s):
        label = hand.gesture if hand and hand.score >= 0.70 else "None"
        action = self.gate.update(label, now_s, self.ALLOWED)
        if action == "Thumb_Up":
            pyautogui.press("right")
            return "Next slide"
        if action == "Thumb_Down":
            pyautogui.press("left")
            return "Previous slide"
        return f"Seen: {label}"
