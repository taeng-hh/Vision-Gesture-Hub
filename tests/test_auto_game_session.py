import unittest
import numpy as np
from core.auto_game_session import AutoGameSession
from modes.mini_game import GalagaGame


class FakeModel:
    classes = [dict(class_id='LEFT', text='왼쪽', game_action='left'),
               dict(class_id='START', text='시작', game_action='start'),
               dict(class_id='OTHER', text='기타', game_action=None)]

    def __init__(self):
        self.label, self.accepted, self.calls = 'LEFT', True, 0

    def predict(self, frames):
        self.calls += 1
        return dict(class_id=self.label, accepted=self.accepted, score=.95)


class TestAutoGameSession(unittest.TestCase):
    def setUp(self):
        self.game = GalagaGame()
        self.game.action('start')
        self.model = FakeModel()
        self.session = AutoGameSession(self.model, self.game)
        self.frame = np.ones((67, 3), dtype=np.float32)

    def feed(self, start, count=10):
        for i in range(count):
            self.session.update(self.frame, start+i*.1)

    def test_moves_without_keys_and_cooldown_prevents_duplicates(self):
        self.feed(0)
        self.assertAlmostEqual(self.game.x, .38)
        calls = self.model.calls
        self.feed(1., 4)
        self.assertEqual(self.model.calls, calls)
        self.assertAlmostEqual(self.game.x, .38)
        self.feed(1.4, 12)
        self.assertAlmostEqual(self.game.x, .26)

    def test_starts_from_ready_automatically(self):
        self.game.reset()
        self.model.label = 'START'
        self.feed(0)
        self.assertEqual(self.game.state, 'running')

    def test_rejected_and_non_game_signs_do_not_move(self):
        self.model.accepted = False
        self.feed(0, 20)
        self.assertNotIn('자동 인식:', self.session.status)
        self.model.accepted = True
        self.model.label = 'OTHER'
        self.feed(2., 20)
        self.assertEqual(self.game.x, .5)
        self.assertIn('자동 인식: 기타', self.session.status)
        self.assertIn('게임 조작 없음', self.session.status)

    def test_pause_and_tracking_loss_discard_old_candidate(self):
        self.feed(0, 6)
        self.game.state = 'paused'
        self.feed(.6)
        self.assertEqual(self.game.x, .5)
        self.game.state = 'running'
        self.feed(1.6, 6)
        self.session.update(None, 2.2)
        self.feed(2.3, 3)
        self.assertEqual(self.game.x, .5)

    def test_game_continues_while_recognizing(self):
        for i in range(10):
            self.session.update(self.frame, i*.1)
            self.game.update(.1)
        self.assertAlmostEqual(self.game.clock, 1.)
        self.assertTrue(self.game.bullets)


if __name__ == '__main__':
    unittest.main()
