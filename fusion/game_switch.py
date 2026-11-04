"""V hand shape held while not recording -> Sign Galaga -> back to the translator.

Same rules as the legacy screen (sign_language_webcam.py): core.GameLaunchGesture
decides when to launch, recording disables it, and a V shown during recording
must be released before it can launch the game.
"""
from pathlib import Path
import cv2
from core.game_launch_gesture import GameLaunchGesture


class GameSwitch:
    detect_interval = 1/12   # Well inside GameLaunchGesture's 0.3 s gap limit.
    detect_width = 640       # The legacy screen ran the recognizer on 640x480.

    def __init__(self, camera, keypoint_model, assets, detector=None, launcher=None, run_game=None):
        self.camera = camera
        self.keypoint_model = Path(keypoint_model)
        self.assets = Path(assets)
        self.launcher = launcher or GameLaunchGesture()
        self._run_game = run_game
        self.detector = detector
        self.error = None
        self.last_detect = None
        if self.detector is None:
            try:
                from core.detector import GestureDetector
                self.detector = GestureDetector(self.assets / 'gesture_recognizer.task')
            except FileNotFoundError:
                self.error = 'models/gesture_recognizer.task 없음 · scripts/setup_assets.py 실행'

    @property
    def progress(self):
        return self.launcher.progress

    def update(self, raw, now, recording):
        """Return True once when the game should open."""
        if self.detector is None:
            return False
        if self.last_detect is not None and now - self.last_detect < self.detect_interval:
            return False
        self.last_detect = now
        h, w = raw.shape[:2]
        small = cv2.resize(raw, (self.detect_width, max(1, round(h*self.detect_width/w)))) if w > self.detect_width else raw
        observation = self.detector.detect(small, int(now*1000))
        return self.launcher.update(observation, now, recording=recording)

    def launch(self):
        """Run the game (blocking); the caller must release its camera first."""
        run_game = self._run_game
        if run_game is None:
            from modes.mini_game import run_game
        try:
            run_game(self.camera, self.keypoint_model, self.assets / 'pose_landmarker_lite.task')
            return '수어 번역으로 돌아왔습니다. Space로 입력하세요.'
        except (FileNotFoundError, RuntimeError) as exc:
            return f'미니게임 실행 실패: {exc}'
        finally:
            self.last_detect = None

    def lines(self, recording):
        if self.error:
            return ['게임 전환 꺼짐: ' + self.error]
        lines = ['녹화하지 않을 때 V 손 모양 1.2초 유지 → 미니게임']
        if self.progress and not recording:
            lines.append(f'미니게임 열기: {int(self.progress * 100)}%')
        return lines

    def close(self):
        if self.detector is not None and hasattr(self.detector, 'close'):
            self.detector.close()
