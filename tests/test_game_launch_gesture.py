import unittest

from core.game_launch_gesture import GameLaunchGesture
from core.types import Hand, Observation


V = Observation((Hand((), 'Victory', .95),))
EMPTY = Observation()


class TestGameLaunchGesture(unittest.TestCase):
    def hold(self, launcher, start=0., recording=False):
        return [launcher.update(V, start+i*.1, recording) for i in range(16)]

    def test_hold_launches_only_once_until_release(self):
        launcher = GameLaunchGesture()
        self.assertEqual(sum(self.hold(launcher)), 1)
        self.assertFalse(any(self.hold(launcher, 2.)))
        launcher.update(EMPTY, 4.)
        self.assertEqual(sum(self.hold(launcher, 4.1)), 1)

    def test_recording_requires_release_afterwards(self):
        launcher = GameLaunchGesture()
        self.assertFalse(any(self.hold(launcher, recording=True)))
        self.assertFalse(any(self.hold(launcher, 2.)))
        launcher.update(EMPTY, 4.)
        self.assertEqual(sum(self.hold(launcher, 4.1)), 1)

    def test_tracking_gap_restarts_hold(self):
        launcher = GameLaunchGesture()
        launcher.update(V, 0.)
        launcher.update(V, .2)
        self.assertFalse(launcher.update(V, 2.))
        self.assertEqual(launcher.progress, 0.)

    def test_wrong_or_uncertain_pose_does_not_launch(self):
        for hand in (Hand((), 'Victory', .5), Hand((), 'Open_Palm', .99)):
            launcher = GameLaunchGesture()
            for i in range(30):
                self.assertFalse(launcher.update(Observation((hand,)), i*.1))


if __name__ == '__main__':
    unittest.main()
