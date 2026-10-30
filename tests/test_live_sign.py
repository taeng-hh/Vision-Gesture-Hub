import unittest
from types import SimpleNamespace
import numpy as np
from core.live_sign import LiveSignSession, MixedLiveSignSession, webcam_frame
from core.types import Hand, Landmark, Observation


def pose_points():
    points = [SimpleNamespace(x=.5, y=.5, visibility=1., presence=1.) for _ in range(33)]
    points[11].x, points[12].x = .7, .3
    points[15].x, points[16].x = .8, .2
    return points


class FakeModel:
    def __init__(self, labels): self.labels = iter(labels)
    def predict(self, frames):
        return dict(label=next(self.labels), distance=.1, margin=.2)


class LiveTests(unittest.TestCase):
    def frame(self):
        pose = pose_points()
        hands = Observation((Hand(tuple(Landmark(.8, .5, -.8) for _ in range(21))),
                             Hand(tuple(Landmark(.2, .5, -.8) for _ in range(21)))))
        return webcam_frame(hands, pose, 640, 480)

    def test_pixels_slots_and_confidence(self):
        f = self.frame()
        self.assertAlmostEqual(f[0, 0], 512)
        self.assertAlmostEqual(f[21, 0], 128)
        self.assertEqual(f[0, 2], 1)  # Not MediaPipe's negative Z.
        self.assertEqual(f[47, 0], 448)  # OpenPose left shoulder.

    def test_missing_pose_and_ambiguous_hands(self):
        self.assertIsNone(webcam_frame(Observation(), None, 640, 480))
        hand = Hand(tuple(Landmark(.5, .5, 0) for _ in range(21)))
        self.assertIsNone(webcam_frame(Observation((hand,)), pose_points(), 640, 480))

    def record(self, session, start):
        session.handle_key(ord(' '), start)
        for i in range(8): session.update(self.frame(), start + i * .05)
        session.handle_key(ord(' '), start + .4)

    def test_sentence_from_words_and_repeat(self):
        session = LiveSignSession(FakeModel(['I', 'HEAD', 'HURT']))
        for start in range(3): self.record(session, start)
        self.assertEqual(session.sentence, '저는 머리가 아파요.')
        session.handle_key(8, 4)
        self.assertEqual(session.tokens, ['I', 'HEAD'])
        session.handle_key(ord('c'), 5)
        self.assertEqual(session.tokens, [])

    def test_no_idle_predictions_and_rejection(self):
        session = LiveSignSession(FakeModel([None]))
        session.update(self.frame(), 0)
        self.assertEqual(session.tokens, [])
        self.record(session, 1)
        self.assertEqual(session.tokens, [])
        self.assertIn('인식 불가', session.status)

    def test_tracking_loss_cancels_and_does_not_add_word(self):
        session = LiveSignSession(FakeModel(['I']))
        session.start(0)
        session.update(self.frame(), .1)
        session.update(None, .2)
        self.assertFalse(session.recording)
        self.assertEqual(session.tokens, [])

    def test_no_hands_finishes_once(self):
        session = LiveSignSession(FakeModel(['I']))
        session.start(0)
        for i in range(8): session.update(self.frame(), i * .05)
        empty = self.frame()
        empty[:42] = 0
        session.update(empty, .4)
        session.update(empty, .8)
        session.update(empty, 1)
        self.assertEqual(session.tokens, ['I'])


if __name__ == '__main__': unittest.main()


class MixedTests(LiveTests):
    def session(self, events):
        class Model:
            classes = [{'class_id': 'WORD0115', 'text': '커피'},
                       {'class_id': 'WORD1589', 'text': '부탁'},
                       {'class_id': 'SEN0354', 'text': '안녕하세요.'}]
            def __init__(self): self.events = iter(events)
            def predict(self, frames): return next(self.events)
        return MixedLiveSignSession(Model())

    def event(self, cid, text, kind='word', accepted=True):
        return dict(accepted=accepted, class_id=cid, text=text, kind=kind, score=.98)

    def test_mixed_words_sentence_and_controls(self):
        s = self.session([self.event('WORD0115', '커피'), self.event('WORD1589', '부탁'),
                          self.event('SEN0354', '안녕하세요.', 'sentence')])
        self.record(s, 0); self.record(s, 1)
        self.assertEqual(s.sentence, '커피를 주세요.')
        self.record(s, 2)
        self.assertEqual(s.sentence, '안녕하세요.')
        self.assertEqual(s.tokens, [])
        s.handle_key(13, 3)
        self.assertEqual(s.sentence, '안녕하세요.')
        s.handle_key(ord('c'), 4)
        self.assertEqual(s.sentence, '')

    def test_rejection_preserves_text(self):
        s = self.session([self.event('WORD0115', '커피'), self.event(None, None, accepted=False)])
        self.record(s, 0); self.record(s, 1)
        self.assertEqual(s.tokens, ['WORD0115'])
        self.assertEqual(s.sentence, '커피')

    def test_sentence_can_exceed_old_limit(self):
        s = self.session([self.event('SEN0354', '안녕하세요.', 'sentence')])
        s.start(0)
        for i in range(160): s.update(self.frame(), i * .05)
        self.assertTrue(s.recording)
        s.handle_key(32, 8)
        self.assertEqual(s.sentence, '안녕하세요.')


class MixedInputRecoveryTests(MixedTests):
    def test_short_tracking_gap_is_not_cancelled(self):
        s=self.session([self.event('WORD0115','커피')]);s.start(0)
        for i in range(8):s.update(self.frame(),i*.05)
        s.update(None,.4);s.update(self.frame(),.6)
        self.assertTrue(s.recording)
        s.handle_key(32,.7)
        self.assertEqual(s.tokens,['WORD0115'])

    def test_long_tracking_gap_cancels(self):
        s=self.session([]);s.start(0);s.update(self.frame(),.1)
        s.update(None,.2);s.update(None,.85)
        self.assertFalse(s.recording)
        self.assertEqual(s.tokens,[])

    def test_hands_out_does_not_end_sentence(self):
        s=self.session([]);s.start(0)
        for i in range(8):s.update(self.frame(),i*.05)
        empty=self.frame();empty[:42]=0
        s.update(empty,.4);s.update(empty,.8)
        self.assertTrue(s.recording)

    def test_timestamp_resampling(self):
        s=self.session([]);s.start(0)
        for i,t in enumerate([0,.1,.2,.6]):
            f=self.frame();f[0,0]=100+t*100;s.update(f,t)
        self.assertAlmostEqual(s.timed_frames()[1,0,0],120,places=3)
