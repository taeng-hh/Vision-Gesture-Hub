
"""Sign-controlled arcade space shooter; never emits operating-system input."""
import math
import random


class GalagaGame:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.state, self.score, self.lives = 'ready', 0, 3
        self.x, self.y, self.stage = .5, .87, 1
        self.clock = self.boost_until = self.invincible_until = 0.
        self.bullets, self.enemy_bullets = [], []
        self.fire_in, self.enemy_fire_in, self.dive_in = 0., 1.5, 3.
        self._formation()

    def _formation(self):
        self.boss = None
        if self.stage == 10:
            self.enemies = []
            self.boss = dict(x=.5, y=.20, hp=60, max_hp=60,
                             half_width=.17, half_height=.075)
            return
        self.enemies = [dict(x=.14+c*.12, y=.13+r*.09, home_x=.14+c*.12,
                             home_y=.13+r*.09, row=r, diving=False, phase=0.)
                        for r in range(3) for c in range(7)]

    def action(self, action):
        if action in ('start', 'run'):
            if self.state in ('ready', 'over'): self.reset()
            self.state = 'running'
        elif self.state == 'running':
            # Preserve action metadata in existing trained model bundles.
            action = {'jump': 'up', 'duck': 'down'}.get(action, action)
            if action == 'left': self.x = max(.04, self.x - .12)
            elif action == 'right': self.x = min(.96, self.x + .12)
            elif action == 'up': self.y = max(.48, self.y - .10)
            elif action == 'down': self.y = min(.94, self.y + .10)
            elif action == 'accelerate': self.boost_until = self.clock + 5.

    def _hit_player(self):
        if self.clock < self.invincible_until: return
        self.lives -= 1
        self.invincible_until = self.clock + 2.
        if self.lives <= 0: self.state = 'over'

    def update(self, dt, frozen=False):
        if frozen or self.state != 'running': return
        dt = max(0., min(dt, .1))
        self.clock += dt
        self.fire_in -= dt
        self.enemy_fire_in -= dt
        self.dive_in -= dt
        if self.fire_in <= 0:
            self.bullets.append(dict(x=self.x, y=self.y-.035))
            self.fire_in = .14 if self.clock < self.boost_until else .32
        if self.boss:
            self.boss['x'] = .5 + .28*math.sin(self.clock*.8)
            if self.enemy_fire_in <= 0:
                for offset in (-.12, -.06, 0., .06, .12):
                    self.enemy_bullets.append(dict(x=self.boss['x']+offset,
                                                  y=self.boss['y']+.075))
                self.enemy_fire_in = .65
        if self.enemies and self.enemy_fire_in <= 0:
            enemy = self.rng.choice(self.enemies)
            self.enemy_bullets.append(dict(x=enemy['x'], y=enemy['y']+.025))
            self.enemy_fire_in = max(.28, 1.3-self.stage*.09)
        if self.enemies and self.dive_in <= 0:
            candidates = [e for e in self.enemies if not e['diving']]
            if candidates:
                enemy = self.rng.choice(candidates)
                enemy['diving'], enemy['phase'] = True, 0.
            self.dive_in = max(.7, 3.-self.stage*.15)
        for enemy in self.enemies:
            if enemy['diving']:
                enemy['phase'] += dt
                enemy['y'] += dt*(.19+self.stage*.015)
                enemy['x'] = max(.03, min(.97, enemy['home_x']+.16*math.sin(enemy['phase']*4)))
                if enemy['y'] > 1.05:
                    enemy['diving'], enemy['y'] = False, enemy['home_y']
            else:
                enemy['x'] = enemy['home_x']+.045*math.sin(self.clock*1.5)
                enemy['y'] = enemy['home_y']
        for bullet in list(self.bullets):
            old_y = bullet['y']
            bullet['y'] -= dt*.85
            if (self.boss and
                    abs(self.boss['x']-bullet['x']) < self.boss['half_width'] and
                    bullet['y']-.025 <= self.boss['y']+self.boss['half_height'] and
                    old_y+.025 >= self.boss['y']-self.boss['half_height']):
                self.bullets.remove(bullet)
                self.boss['hp'] -= 1
                if self.boss['hp'] <= 0:
                    self.score += 5000
                    self.boss = None
                continue
            target = next((e for e in self.enemies if abs(e['x']-bullet['x']) < .033
                           and bullet['y']-.025 <= e['y'] <= old_y+.025), None)
            if target:
                self.enemies.remove(target)
                self.bullets.remove(bullet)
                self.score += 200 if target['diving'] else 100
            elif bullet['y'] < 0: self.bullets.remove(bullet)
        for bullet in list(self.enemy_bullets):
            old_y = bullet['y']
            bullet['y'] += dt*(.28+self.stage*.015)
            if abs(bullet['x']-self.x) < .026 and old_y-.025 <= self.y <= bullet['y']+.025:
                self.enemy_bullets.remove(bullet)
                self._hit_player()
            elif bullet['y'] > 1.05: self.enemy_bullets.remove(bullet)
        for enemy in self.enemies:
            if abs(enemy['x']-self.x) < .04 and abs(enemy['y']-self.y) < .045:
                self._hit_player()
        if not self.enemies and self.boss is None and self.state == 'running':
            self.stage += 1
            self.enemy_bullets.clear()
            self.dive_in, self.enemy_fire_in = 2., 1.
            self._formation()


RunnerGame = GalagaGame


def run_game(camera=0, model=None, pose_model=None, window=None):
    """window: reuse this open window (the translator's) instead of opening one."""
    from contextlib import ExitStack
    from pathlib import Path
    import time
    import cv2
    import mediapipe as mp
    from core.detector import GestureDetector
    from core.sign_language_session import webcam_frame
    from core.auto_game_session import AutoGameSession
    from ksl.predict import Recognizer
    from ui import theme
    from ui.mini_game_view import draw_galaga

    root = Path(__file__).resolve().parents[1]
    model = Path(model) if model else root / 'models/ksl30_robust_v2'
    pose_model = Path(pose_model) if pose_model else root / 'models/pose_landmarker_lite.task'
    if not (model / 'model.pt').is_file() or not pose_model.is_file():
        raise FileNotFoundError(f'Required models: {model / "model.pt"}, {pose_model}')
    game = GalagaGame()

    session = AutoGameSession(Recognizer(model), game)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_model.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    title, cap = window or theme.WINDOW_TITLE, None
    try:
        with ExitStack() as stack:
            detector = stack.enter_context(GestureDetector(root / 'models/gesture_recognizer.task'))
            pose = stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
            cap = cv2.VideoCapture(camera)
            if not cap.isOpened(): raise RuntimeError('Camera unavailable')
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if window is None:
                cv2.resizeWindow(title, theme.WIDTH, theme.HEIGHT)
            previous, timestamp = time.monotonic(), -1
            while True:
                ok, raw = cap.read()
                if not ok: break
                now = time.monotonic()
                timestamp = max(timestamp + 1, int(now * 1000))
                observation = detector.detect(raw, timestamp)
                body = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)), timestamp)
                h, w = raw.shape[:2]
                skeleton = webcam_frame(observation,
                    body.pose_landmarks[0] if body.pose_landmarks else None, w, h)
                session.update(skeleton, now)
                game.update(now - previous, frozen=skeleton is None)
                previous = now
                canvas = draw_galaga(game, raw, session, skeleton is not None)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q'), 27): break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1: break
                if key == ord('r'):
                    game.reset()
                    session.cancel_recording()
                elif key == ord('p'):
                    if game.state == 'running': game.state = 'paused'
                    elif game.state == 'paused': game.state = 'running'
                    session.cancel_recording()
                else:
                    action = {13: 'start', 10: 'start', ord('a'): 'left', ord('d'): 'right',
                              ord('w'): 'up', ord('s'): 'down', ord('f'): 'accelerate'}.get(key)
                    if action: game.action(action)
    finally:
        if cap is not None: cap.release()
        if window is None:
            try: cv2.destroyWindow(title)
            except cv2.error: pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model')
    parser.add_argument('--pose-model')
    args = parser.parse_args()
    run_game(args.camera, args.model, args.pose_model)
