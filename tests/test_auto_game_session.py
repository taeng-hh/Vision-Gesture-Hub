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

    def test_game_scope_is_used_when_available(self):
        calls=[]
        def predict_game(frames):
            calls.append(1)
            return dict(class_id='LEFT',accepted=True,score=.95)
        self.model.predict_game=predict_game
        self.feed(0)
        self.assertTrue(calls)
        self.assertEqual(self.model.calls,0)
        self.assertAlmostEqual(self.game.x,.38)

    def test_multiscale_prefers_supported_window_and_compares_same_input(self):
        batches=[]
        def compare_windows(clips):
            batches.append([c.copy() for c in clips])
            pairs=[]
            for i,c in enumerate(clips):
                strength=.8+i*.02
                event=dict(class_id='LEFT',candidate_id='LEFT',accepted=True,score=.96,global_score=strength)
                pairs.append(dict(normal=dict(event),game=dict(event)))
            return pairs
        self.model.compare_windows=compare_windows
        self.feed(0,20)
        self.assertTrue(batches)
        self.assertTrue(all(c.shape==(32,67,3) for b in batches for c in b))
        self.assertAlmostEqual(self.game.x,.38)
        self.assertEqual(self.session.last_comparison['normal']['candidate_id'],'LEFT')
        self.assertIn('실행',self.session.status)

    def test_multiscale_rejected_windows_do_not_trigger_action(self):
        def compare_windows(clips):
            event=dict(class_id=None,candidate_id='LEFT',accepted=False,score=.90,global_score=.05)
            return [dict(normal=dict(event),game=dict(event)) for c in clips]
        self.model.compare_windows=compare_windows
        self.feed(0,20)
        self.assertAlmostEqual(self.game.x,.5)
        self.assertIn('보류',self.session.status)

    def test_short_partial_prediction_waits_for_context(self):
        def compare_windows(clips):
            event=dict(class_id='LEFT',candidate_id='LEFT',accepted=True,score=.96,global_score=.96,margin=.8)
            return [dict(normal=dict(event),game=dict(event)) for c in clips]
        self.model.compare_windows=compare_windows
        self.feed(0,12)
        self.assertAlmostEqual(self.game.x,.5)
        self.assertIn('문맥 대기',self.session.status)
        self.feed(1.2,4)
        self.assertAlmostEqual(self.game.x,.38)

    def test_full_and_game_very_high_agreement_executes_without_second_hit(self):
        def compare_windows(clips):
            event=dict(class_id='LEFT',candidate_id='LEFT',accepted=True,score=.999,global_score=.999,margin=.99)
            return [dict(normal=dict(event),game=dict(event)) for c in clips]
        self.model.compare_windows=compare_windows
        self.feed(0,6)
        self.assertAlmostEqual(self.game.x,.38)

    def test_prediction_schedule_does_not_drift_at_30fps(self):
        times=[]
        def reject(frames):
            times.append(self.session.last_predict)
            return dict(class_id=None,accepted=False,score=.1)
        self.model.predict=reject
        for i in range(42):self.session.update(self.frame,i/30)
        np.testing.assert_allclose(times,[.5,.7,.9,1.1,1.3],atol=1e-6)

    def test_async_pending_preserves_but_does_not_increment_confirmation(self):
        class Assist:
            enabled=True
            pending=object()
            calls=0
            def invalidate(self):self.pending=None
            def evaluate(self,pair,samples,now,window_seconds):
                self.calls+=1;self.pending=object()
                if self.calls%2==1:
                    pair=dict(pair);pair['game']=dict(pair['game'],class_id='LEFT',accepted=True)
                return pair
        assist=Assist();self.session=AutoGameSession(self.model,self.game,assist)
        def compare_windows(clips):
            normal=dict(class_id=None,candidate_id='LEFT',accepted=False,score=.5,global_score=.5,margin=.2)
            game=dict(normal,scope='game')
            return [dict(normal=dict(normal),game=dict(game)) for c in clips]
        self.model.compare_windows=compare_windows
        self.feed(0,6)
        self.assertEqual(self.session.matches,1)
        self.feed(.6,2)
        self.assertEqual(self.session.matches,1)
        self.feed(.8,8)
        self.assertAlmostEqual(self.game.x,.38)

if __name__ == '__main__':
    unittest.main()
