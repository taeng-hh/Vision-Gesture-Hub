"""Debounce frame-level predictions and require neutral before retriggering."""


class OneShotGestureGate:
    def __init__(self, stable_frames=8, neutral_frames=3, cooldown_s=1.0):
        self.stable_frames = stable_frames
        self.neutral_frames = neutral_frames
        self.cooldown_s = cooldown_s
        self.reset()

    def reset(self):
        self.previous = None
        self.count = 0
        self.neutral_count = 0
        self.armed = True
        self.last_trigger = float("-inf")

    def update(self, label: str, now_s: float, allowed: set[str]) -> str | None:
        if label not in allowed:
            self.neutral_count += 1
            self.count = 0
            self.previous = None
            if self.neutral_count >= self.neutral_frames:
                self.armed = True
            return None

        self.neutral_count = 0
        if label == self.previous:
            self.count += 1
        else:
            self.previous = label
            self.count = 1

        if (self.armed and self.count >= self.stable_frames
                and now_s - self.last_trigger >= self.cooldown_s):
            self.armed = False
            self.last_trigger = now_s
            return label
        return None
