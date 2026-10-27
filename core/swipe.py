"""Detect a right-to-left open-palm swipe in mirrored camera coordinates."""
from core.geometry import is_open_palm


class OpenPalmSwipe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self._start = None
        self._last_time = None
        self._handedness = None
        self._neutral_since = None
        self._armed = True
        self._last_trigger = float("-inf")
        self._last_open = float("-inf")

    def update(self, hand, now_s):
        detected = bool(hand and len(hand.landmarks) >= 21
                        and ((hand.gesture == "Open_Palm" and hand.score >= 0.7)
                             or is_open_palm(hand)))
        if detected:
            self._last_open = now_s
        # Brief label/tracking dropouts pause the trajectory, rather than cancel it.
        self.active = detected or bool(hand and self.active and now_s - self._last_open <= 0.12)
        if self.active and not detected:
            return False
        if not self.active:
            self._start = None
            self._last_time = None
            self._handedness = None
            if self._neutral_since is None:
                self._neutral_since = now_s
            if now_s - self._neutral_since >= 0.15:
                self._armed = True
            return False

        self._neutral_since = None
        if (self._last_time is not None and now_s - self._last_time > 0.25
                or self._handedness != hand.handedness):
            self._start = None
        self._last_time = now_s
        self._handedness = hand.handedness
        points = [hand.landmarks[i] for i in (0, 5, 9, 13, 17)]
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        # Returning to the right rearms the next swipe without folding the hand.
        if not self._armed and x >= 0.65 and now_s - self._last_trigger >= 0.8:
            self._armed = True
        if not self._armed or now_s - self._last_trigger < 0.8:
            return False
        if self._start is not None:
            sx, sy, started = self._start
            elapsed = now_s - started
            if elapsed > 0.8 or abs(y - sy) > 0.18 or x > sx + 0.08:
                self._start = None
            elif 0.08 <= elapsed and sx - x >= 0.22:
                self._armed = False
                self._last_trigger = now_s
                self._start = None
                return True
        if self._start is None and x >= 0.55:
            self._start = (x, y, now_s)
        return False
