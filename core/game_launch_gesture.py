"""Debounced game shortcut, disabled while recording sign language."""


class GameLaunchGesture:
    def __init__(self, hold_seconds=1.2):
        self.hold_seconds = hold_seconds
        self.since = self.last_time = None
        self.latched = False
        self.progress = 0.

    def update(self, observation, now, recording=False):
        active = any(h.gesture == 'Victory' and h.score >= .8
                     for h in observation.hands)
        gap = self.last_time is not None and now - self.last_time > .3
        self.last_time = now
        if recording or not active:
            self.since, self.progress = None, 0.
            # A pose held during recording must be released before launching.
            self.latched = active
            return False
        if self.since is None or gap:
            self.since = now
        self.progress = min(1., max(0., (now-self.since)/self.hold_seconds))
        if self.progress >= 1. and not self.latched:
            self.latched = True
            return True
        return False
