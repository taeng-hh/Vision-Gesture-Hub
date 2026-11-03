"""Continuous, bounded webcam windows for in-game sign commands."""
from collections import deque
import numpy as np


class AutoGameSession:
    window_seconds = 1.6
    min_seconds = .5
    predict_interval = .2
    cooldown_seconds = .8

    def __init__(self, model, game):
        self.model, self.game = model, game
        self.classes = {c['class_id']: c for c in model.classes}
        self.status = '자동 인식 중: 시작 수어 또는 Enter로 출격하세요.'
        self.cancel_recording()

    def cancel_recording(self):
        self.frames = deque(maxlen=120)
        self.last_time = self.last_hand = self.last_predict = None
        self.cooldown_until = 0.
        self.candidate, self.matches = None, 0

    def update(self, frame, now):
        if self.last_time is not None and (now <= self.last_time or now-self.last_time > .6):
            self.cancel_recording()
        self.last_time = now
        if self.game.state == 'paused':
            self.frames.clear()
            self.candidate, self.matches = None, 0
            self.status = '일시정지 중 · P로 계속'
            return
        if frame is None or not np.any(frame[:42, 2] >= .2):
            self.candidate, self.matches = None, 0
            if frame is None or self.last_hand is None or now-self.last_hand >= .3:
                self.frames.clear()
            self.status = '자동 인식 대기: 어깨와 손을 화면에 보여 주세요.'
            return
        self.last_hand = now
        if now < self.cooldown_until:
            return
        self.frames.append((now, frame.copy()))
        while self.frames and now-self.frames[0][0] > self.window_seconds:
            self.frames.popleft()
        if len(self.frames) < 4 or now-self.frames[0][0] < self.min_seconds:
            return
        if self.last_predict is not None and now-self.last_predict < self.predict_interval:
            return
        self.last_predict = now
        times = np.asarray([t for t, _ in self.frames])
        frames = np.asarray([f for _, f in self.frames])
        target = np.linspace(times[0], times[-1], len(times))
        flat = frames.reshape(len(frames), -1)
        clip = np.stack([np.interp(target, times, c) for c in flat.T], axis=1).reshape(frames.shape)
        try:
            result = self.model.predict(clip)
        except ValueError:
            self.frames.clear()
            self.candidate, self.matches = None, 0
            self.status = '좌표 추적 대기: 손과 어깨를 화면에 유지하세요.'
            return
        entry = self.classes.get(result.get('class_id'), {})
        action = entry.get('game_action') if result.get('accepted') else None
        if not action:
            self.candidate, self.matches = None, 0
            if result.get('accepted') and entry:
                self.status = f"자동 인식: {entry['text']} · 점수 {result['score']:.2f} · 게임 조작 없음"
            else:
                self.status = '자동 인식 중: 게임 조작 수어를 보여 주세요.'
            return
        self.matches = self.matches + 1 if action == self.candidate else 1
        self.candidate = action
        if self.matches < 2:
            return
        self.game.action(action)
        self.status = f"자동 인식: {entry['text']} · 점수 {result['score']:.2f}"
        self.frames.clear()
        self.candidate, self.matches = None, 0
        self.cooldown_until = now + self.cooldown_seconds
