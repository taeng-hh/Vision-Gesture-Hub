"""Virtual pointer: index tip moves cursor; rising pinch edge clicks once."""
import pyautogui
from core.geometry import normalize_to_roi, pinch_ratio


class PCController:
    def __init__(self):
        self._position = None
        self._pinched = False
        self._ready = False
        self._last_click = float("-inf")
        pyautogui.FAILSAFE = True
        self._screen_w, self._screen_h = pyautogui.size()

    def reset(self):
        self._position = None
        self._pinched = False
        self._ready = False

    def update(self, hand, now_s):
        if hand is None:
            self.reset()
            return "No hand"
        ratio = pinch_ratio(hand)
        if ratio is None:
            return "No landmarks"

        # Approximate pointer mapping (not physical touch on the TV screen).
        index = hand.landmarks[8]
        x, y = normalize_to_roi(index.x, index.y)
        target_x = 5 + x * (self._screen_w - 10)
        target_y = 5 + y * (self._screen_h - 10)
        alpha = 0.28
        if self._position is None:
            self._position = (target_x, target_y)
        else:
            px, py = self._position
            self._position = (px + alpha * (target_x-px), py + alpha * (target_y-py))
        pyautogui.moveTo(*self._position, duration=0)

        # Hysteresis: must open pinch before first click or after reset.
        if ratio > 0.65:
            self._pinched = False
            self._ready = True
            return "Move / ready"
        if ratio < 0.38 and not self._pinched and self._ready:
            self._pinched = True
            self._ready = False
            if now_s - self._last_click >= 0.6:
                pyautogui.click()
                self._last_click = now_s
                return "Click"
        return f"Pointer / pinch {ratio:.2f}"
