"""Camera-coordinate navigation; never emits OS input."""
from dataclasses import dataclass

from core.mode_manager import Mode


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    bounds: tuple[float, float, float, float]


CARDS = (
    Target(Mode.SIGN.value, "SIGN LANGUAGE", (0.04, 0.40, 0.32, 0.80)),
    Target(Mode.PRESENTATION.value, "PRESENTATION", (0.36, 0.40, 0.64, 0.80)),
    Target(Mode.PC.value, "PC CONTROL", (0.68, 0.40, 0.96, 0.80)),
)
HOME = Target("HOME", "HOME", (0.02, 0.02, 0.27, 0.15))
TOGGLE = Target("TOGGLE", "ENABLE / DISABLE", (0.60, 0.02, 0.98, 0.15))


def targets_for(mode):
    if mode == Mode.IDLE:
        return CARDS
    return (HOME, TOGGLE) if mode in (Mode.PC, Mode.PRESENTATION) else (HOME,)


class GestureNavigation:
    def __init__(self, dwell_s=0.8):
        self.dwell_s = dwell_s
        self.reset()

    def reset(self):
        self.hover = None
        self.progress = 0.0
        self.pointer = None
        self._since = None
        self._last_time = None
        # After activation the finger must leave the controls to select again.
        self._latched = False

    def update(self, hand, mode, now_s):
        self.pointer = None
        target = None
        if hand is not None and len(hand.landmarks) >= 21:
            p = hand.landmarks[8]
            self.pointer = (p.x, p.y)
            target = next((t for t in targets_for(mode)
                           if t.bounds[0] <= p.x <= t.bounds[2]
                           and t.bounds[1] <= p.y <= t.bounds[3]), None)
        # A stalled camera must not count as continuous dwell.
        gap = self._last_time is not None and now_s - self._last_time > 0.3
        self._last_time = now_s
        if target is None:
            self.hover = None
            self._since = None
            self.progress = 0.0
            self._latched = False
            return None
        if target != self.hover or gap:
            self.hover = target
            self._since = now_s
        self.progress = min(1.0, max(0.0, (now_s - self._since) / self.dwell_s))
        if self.progress >= 1 and not self._latched:
            self._latched = True
            return target.key
        return None
