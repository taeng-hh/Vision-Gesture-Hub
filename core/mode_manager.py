"""One active mode at a time, with OS actions disabled at startup."""
from enum import Enum


class Mode(str, Enum):
    IDLE = "IDLE"
    SIGN = "SIGN"
    PC = "PC"
    PRESENTATION = "PRESENTATION"


KEY_MODES = {
    ord("0"): Mode.IDLE,
    ord("1"): Mode.SIGN,
    ord("2"): Mode.PC,
    ord("3"): Mode.PRESENTATION,
}


class ModeManager:
    def __init__(self):
        self.mode = Mode.IDLE
        self.actions_enabled = False

    def select(self, mode: Mode):
        self.mode = mode
        self.actions_enabled = False  # Safe on every mode transition.

    def toggle_actions(self):
        if self.mode not in (Mode.PC, Mode.PRESENTATION):
            self.actions_enabled = False
        else:
            self.actions_enabled = not self.actions_enabled
        return self.actions_enabled
