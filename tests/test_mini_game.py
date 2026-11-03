import random
import unittest
from modes.mini_game import GalagaGame


class TestGalagaGame(unittest.TestCase):
    def test_direction_sign_metadata_moves_player(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'config/ksl_classes.json',
                     root / 'models/ksl30_robust_v2/classes.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            classes = data['classes'] if isinstance(data, dict) else data
            by_id = {c['class_id']: c for c in classes}
            for class_id, action, dx, dy in (
                ('WORD1342', 'left', -.12, 0),
                ('WORD1343', 'right', .12, 0),
                ('WORD2100', 'up', 0, -.10),
                ('WORD2577', 'down', 0, .10),
            ):
                with self.subTest(path=path, action=action):
                    game = self.game()
                    game.x = game.y = .6
                    self.assertEqual(by_id[class_id]['game_action'], action)
                    game.action(by_id[class_id]['game_action'])
                    self.assertAlmostEqual(game.x, .6 + dx)
                    self.assertAlmostEqual(game.y, .6 + dy)

    def game(self):
        game = GalagaGame(random.Random(0))
        game.action('start')
        return game

    def test_four_directions_and_model_aliases(self):
        game = self.game()
        for action, attr, boundary in [('left', 'x', .04), ('right', 'x', .96),
                                       ('jump', 'y', .48), ('duck', 'y', .94)]:
            for _ in range(30): game.action(action)
            self.assertAlmostEqual(getattr(game, attr), boundary)
        game.action('up')
        self.assertAlmostEqual(game.y, .84)
        game.action('down')
        self.assertAlmostEqual(game.y, .94)

    def test_recording_and_pause_freeze_simulation(self):
        game = self.game()
        game.action('accelerate')
        before = repr(game.__dict__)
        game.update(10, frozen=True)
        self.assertEqual(repr(game.__dict__), before)
        game.state = 'paused'
        before = repr(game.__dict__)
        game.update(.1)
        self.assertEqual(repr(game.__dict__), before)

    def test_auto_fire_and_rapid_fire(self):
        game = self.game()
        game.update(.1)
        self.assertEqual(len(game.bullets), 1)
        normal = game.fire_in
        game.action('accelerate')
        game.fire_in = 0
        game.update(.1)
        self.assertLess(game.fire_in, normal)

    def test_shooting_scores_and_advances_stage(self):
        for diving, points in [(False, 100), (True, 200)]:
            game = self.game()
            enemy = game.enemies[0]
            enemy.update(x=.5, home_x=.5, y=.3, home_y=.3, diving=diving, phase=-.1)
            game.enemies = [enemy]
            game.bullets = [dict(x=.5, y=.36)]
            game.update(.1)
            self.assertEqual(game.score, points)
            self.assertEqual(game.stage, 2)
            self.assertEqual(len(game.enemies), 21)

    def test_damage_invincibility_game_over_and_restart(self):
        game = self.game()
        for expected in (2, 1, 0):
            game.invincible_until = 0
            game.enemy_bullets = [dict(x=game.x, y=game.y-.01)]
            game.update(.1)
            self.assertEqual(game.lives, expected)
            if expected:
                game.enemy_bullets = [dict(x=game.x, y=game.y-.01)]
                game.update(.1)
                self.assertEqual(game.lives, expected)
        self.assertEqual(game.state, 'over')
        game.action('start')
        self.assertEqual((game.state, game.lives, game.score, game.stage), ('running', 3, 0, 1))

    def test_enemy_fire_and_dive(self):
        game = self.game()
        game.enemy_fire_in = game.dive_in = 0
        game.update(.1)
        self.assertEqual(len(game.enemy_bullets), 1)
        self.assertTrue(any(e['diving'] for e in game.enemies))

    def test_enemy_body_collision(self):
        game = self.game()
        enemy = game.enemies[0]
        enemy.update(x=game.x, y=game.y, home_x=game.x, home_y=game.y)
        game.update(.01)
        self.assertEqual(game.lives, 2)

    def test_ready_and_unknown_actions(self):
        game = GalagaGame()
        game.action('unknown')
        game.update(.1)
        self.assertEqual(game.clock, 0)
        self.assertEqual(game.state, 'ready')

    def test_stage_ten_spawns_boss(self):
        game = self.game()
        game.stage = 9
        game.enemies.clear()
        game.update(.01)
        self.assertEqual(game.stage, 10)
        self.assertEqual(game.enemies, [])
        self.assertEqual(game.boss['hp'], 60)
        game.enemy_fire_in = 0
        game.update(.01)
        self.assertEqual(len(game.enemy_bullets), 5)
        self.assertEqual(game.stage, 10)

    def test_boss_damage_defeat_and_restart(self):
        game = self.game()
        game.stage = 10
        game._formation()
        game.bullets = [dict(x=.5, y=.30)]
        game.update(.1)
        self.assertEqual(game.boss['hp'], 59)
        game.boss['hp'] = 1
        game.bullets = [dict(x=game.boss['x'], y=.30)]
        game.update(.1)
        self.assertEqual(game.score, 5000)
        self.assertEqual(game.stage, 11)
        self.assertIsNone(game.boss)
        self.assertEqual(len(game.enemies), 21)
        game.reset()
        self.assertIsNone(game.boss)
        self.assertEqual(game.stage, 1)


if __name__ == '__main__':
    unittest.main()
