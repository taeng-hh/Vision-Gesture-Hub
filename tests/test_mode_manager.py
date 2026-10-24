import unittest
from core.mode_manager import ModeManager, Mode


class TestModeManager(unittest.TestCase):
    def test_actions_off_on_transitions(self):
        mm = ModeManager()
        self.assertFalse(mm.actions_enabled)
        mm.select(Mode.PC)
        self.assertTrue(mm.toggle_actions())
        mm.select(Mode.PRESENTATION)
        self.assertFalse(mm.actions_enabled)
        mm.select(Mode.SIGN)
        self.assertFalse(mm.toggle_actions())
