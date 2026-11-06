import unittest
import numpy as np
from ksl.predict import prediction_event

class GamePredictionTests(unittest.TestCase):
    classes=[dict(class_id='LEFT',kind='word',text='왼쪽',game_action='left'),
             dict(class_id='RIGHT',kind='word',text='오른쪽',game_action='right'),
             dict(class_id='FRIEND',kind='word',text='친구',game_action=None),
             dict(class_id='HELLO',kind='sentence',text='안녕하세요',game_action=None)]
    def test_normal_prediction_is_still_full_vocabulary(self):
        event=prediction_event(self.classes,[.02,.01,.96,.01])
        self.assertTrue(event['accepted']);self.assertEqual(event['class_id'],'FRIEND')
    def test_competitive_game_evidence_can_win_in_game_scope(self):
        scores=[.39,.02,.40,.19]
        self.assertFalse(prediction_event(self.classes,scores)['accepted'])
        event=prediction_event(self.classes,scores,game_only=True)
        self.assertTrue(event['accepted']);self.assertEqual(event['class_id'],'LEFT')
        self.assertAlmostEqual(event['global_score'],.39)
    def test_strong_friend_is_not_forced_into_direction(self):
        event=prediction_event(self.classes,[.025,.005,.96,.01],game_only=True)
        self.assertFalse(event['accepted']);self.assertIsNone(event['game_action'])
    def test_uniform_or_ambiguous_game_scores_are_rejected(self):
        for scores in [[.25,.25,.25,.25],[.42,.40,.10,.08]]:
            self.assertFalse(prediction_event(self.classes,scores,game_only=True)['accepted'])
    def test_small_absolute_game_probability_is_rejected(self):
        event=prediction_event(self.classes,[.31,.01,.34,.34],game_only=True)
        self.assertFalse(event['accepted'])
    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):prediction_event(self.classes,[np.nan,0,0,1],game_only=True)
