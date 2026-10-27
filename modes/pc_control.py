"""Virtual pointer with motion smoothing and debounced pinch clicks."""
from math import exp, hypot

import pyautogui
from core.geometry import normalize_to_roi, pinch_ratio
from core.swipe import OpenPalmSwipe


class PCController:
    def __init__(self, *, backend=None, pinch_close=0.45, pinch_open=0.55):
        if not 0 < pinch_close < pinch_open:
            raise ValueError("Expected 0 < pinch_close < pinch_open")
        self._mouse = backend if backend is not None else pyautogui
        self._mouse.FAILSAFE = True
        self._screen_w, self._screen_h = self._mouse.size()
        self.pinch_close = pinch_close
        self.pinch_open = pinch_open
        self._last_click = float("-inf")
        self._swipe = OpenPalmSwipe()
        self.reset()

    def reset(self):
        self._swipe.reset()
        self._position = None
        self._filtered = None
        self._last_update = None
        self._pinch_since = None
        self._open_since = None
        self._pinched = False
        self._ready = False

    def update(self, hand, now_s):
        switched = self._swipe.update(hand, now_s)
        if self._swipe.active:
            # Keep pointer and pinch state out of an app-switch gesture.
            self._position = None
            self._filtered = None
            self._last_update = None
            self._pinch_since = None
            self._open_since = None
            self._pinched = False
            self._ready = False
            if switched:
                self._mouse.hotkey("alt", "tab")
                return "App switched: Alt+Tab"
            return "Open palm: swipe right to left to switch app"
        ratio = pinch_ratio(hand, finger_tip=12)
        if ratio is None:
            self.reset()
            return "No hand / landmarks"

        dt = (1 / 30 if self._last_update is None
              else max(0.0, min(now_s - self._last_update, 0.1)))
        self._last_update = now_s

        # Require a deliberate open hand before arming, including after reset.
        if ratio >= self.pinch_open:
            self._pinch_since = None
            if self._open_since is None:
                self._open_since = now_s
            if now_s - self._open_since >= 0.10:
                self._pinched = False
                self._ready = True
        else:
            self._open_since = None

        clicked = False
        if ratio <= self.pinch_close:
            if self._pinch_since is None:
                self._pinch_since = now_s
            if (self._ready and not self._pinched
                    and now_s - self._pinch_since >= 0.08):
                self._pinched = True
                self._ready = False
                if now_s - self._last_click >= 0.6:
                    self._mouse.click()
                    self._last_click = now_s
                    clicked = True
        else:
            self._pinch_since = None

        # Freeze while pinching: closing fingers must not move the click target.
        if ratio > self.pinch_close:
            index = hand.landmarks[8]
            x, y = normalize_to_roi(index.x, index.y)
            target = (5 + x * (self._screen_w - 10),
                      5 + y * (self._screen_h - 10))
            if self._filtered is None:
                self._filtered = target
            else:
                fx, fy = self._filtered
                # Slow small movements; follow intentional large moves faster.
                distance = hypot(target[0] - fx, target[1] - fy)
                tau = 0.12 if distance < 80 else 0.045
                alpha = 1 - exp(-dt / tau)
                self._filtered = (fx + alpha * (target[0] - fx),
                                  fy + alpha * (target[1] - fy))
            if (self._position is None
                    or hypot(self._filtered[0] - self._position[0],
                             self._filtered[1] - self._position[1]) >= 4):
                self._position = self._filtered
                self._mouse.moveTo(*self._position, duration=0)

        state = "READY" if self._ready else "Open thumb + middle to arm"
        if self._pinched:
            state = "Release pinch"
        if clicked:
            state = "Click"
        return f"{state} | T+M {ratio:.2f} (close {self.pinch_close:.2f})"
