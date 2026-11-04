"""Render the translator and game screens in fixed states to PNG (no webcam).

    python scripts/render_ui_preview.py            # writes docs/ui/*.png

The camera area uses a synthetic frame with a fake skeleton, so no person or
room appears in the images.
"""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusion.preview import TrackingSnapshot, draw_skeleton  # noqa: E402
from ui.translator_view import TranslatorState, draw_translator  # noqa: E402


def synthetic_camera(width=1280, height=720):
    """Dim gradient 'room' plus a fake upper-body and two-hand skeleton."""
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    base = (38 + 26*y + 10*x).astype(np.uint8)
    frame = np.dstack([base+18, base+8, base]).astype(np.uint8)
    cv2.ellipse(frame, (width//2, height+40), (300, 260), 0, 180, 360, (70, 62, 58), -1)   # Torso.
    cv2.circle(frame, (width//2, height//2-70), 95, (92, 108, 132), -1)                      # Head.

    def hand(cx, cy, spread):
        points = [(cx, cy)]
        for finger in range(5):
            angle = -2.4 + finger*.45*spread
            for joint in range(1, 5):
                points.append((cx+np.cos(angle)*joint*.022, cy+np.sin(angle)*joint*.03))
        return tuple(points)

    pose = [(.5, .5, 0.)]*33
    pose[11], pose[12] = (.36, .78, 1.), (.64, .78, 1.)
    pose[13], pose[14] = (.30, .95, 1.), (.70, .95, 1.)
    pose[15], pose[16] = (.33, .62, 1.), (.67, .62, 1.)
    pose[23], pose[24] = (.40, 1.2, 0.), (.60, 1.2, 0.)
    snapshot = TrackingSnapshot(1., (hand(.33, .60, 1.), hand(.67, .60, -1.)), tuple(pose), True, True)
    display = cv2.flip(frame, 1)
    draw_skeleton(display, snapshot, 1., mirror=True)
    return display


def states(camera):
    rois = np.stack([np.full((224, 224, 3), v, np.uint8) for v in (90, 120, 150)])
    base = dict(camera=camera, now=10., hands=2, shoulders=True, keypoints=True, game_progress=0.)
    word = dict(accepted=True, kind='word', text='행복', score=.91, candidates=[('행복', .91), ('좋다', .04)])
    return {
        'translator_ready': TranslatorState(**base, message='Space로 시작하고 단어 또는 문장 하나를 수행하세요.'),
        'translator_recording': TranslatorState(**{**base, 'game_progress': 0.}, status='recording', elapsed=1.4,
                                                message='입력 중: 동작을 마치면 Space를 누르세요.', tokens=('나',), sentence='나'),
        'translator_analyzing': TranslatorState(**base, status='analyzing', message='키포인트+CNN보조 분석 중입니다.',
                                                tokens=('나',), sentence='나'),
        'translator_word': TranslatorState(**{**base, 'game_progress': .55}, event=word, event_age=5., tokens=('나', '행복'),
                                           sentence='저는 행복해요.', history=('안녕하세요.', '밥을 주세요.'),
                                           recent=(('행복', .91), ('나', .88), ('밥을 주세요.', .83)),
                                           message='키포인트+CNN보조: 행복 · 0.91 · 분석 0.18초'),
        'translator_word_flash': TranslatorState(**base, event=word, event_age=.05, tokens=('나', '행복'), sentence='저는 행복해요.',
                                                 recent=(('행복', .91), ('나', .88)),
                                                 message='키포인트+CNN보조: 행복 · 0.91 · 분석 0.18초'),
        'translator_retry': TranslatorState(**{**base, 'hands': 1, 'keypoints': False}, tracking_note='손과 양어깨 위치를 함께 확인하세요',
                                            event=dict(accepted=False, kind=None, text='감사', score=.58,
                                                       candidates=[('감사', .58), ('죄송', .31), ('부탁', .05)]),
                                            message='키포인트+CNN보조 인식 보류: 감사 · 점수 0.58 · 차이 0.27',
                                            tokens=('커피',), sentence='커피'),
        'translator_debug': TranslatorState(**base, debug=True, event=word, tokens=('나', '행복'), sentence='저는 행복해요.',
                                            debug_lines=('판정 모드: 키포인트+CNN보조 / 30클래스', '출력 기준: 점수 0.75 / 차이 0.15',
                                                         '키포인트: 행복 0.71', '영상: 행복 0.88', 'CNN 후보 일치 확인'),
                                            rois=rois, message='키포인트+CNN보조: 행복 · 0.91 · 분석 0.62초'),
        'translator_game_handoff': TranslatorState(**{**base, 'game_progress': 1.}, tokens=('나',), sentence='나',
                                                   overlay=('SIGN GALAGA로 이동합니다', '게임에서 Q 또는 Esc를 누르면 번역 화면으로 돌아옵니다')),
        'translator_reconnecting': TranslatorState(**{**base, 'hands': 0, 'shoulders': False, 'keypoints': False},
                                                   status='reconnecting', camera_note='카메라 다시 연결 중…',
                                                   message='카메라 입력이 끊겨 녹화를 취소했습니다. Space로 다시 시작하세요.'),
    }


def galaga(camera):
    from modes.mini_game import GalagaGame
    from ui.mini_game_view import draw_galaga
    game = GalagaGame(rng=random.Random(3))
    game.action('start')
    for _ in range(40):
        game.update(.05)
    return draw_galaga(game, cv2.flip(camera, 1), SimpleNamespace(status='자동 인식: 왼쪽 · 점수 0.88'), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/ui')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    camera = synthetic_camera()
    images = {name: draw_translator(state) for name, state in states(camera).items()}
    images['galaga'] = galaga(camera)
    for name, image in images.items():
        path = args.out/f'{name}.png'
        # imencode + write_bytes keeps Korean/Unicode paths working on Windows.
        ok, data = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError(f'PNG encode failed: {name}')
        path.write_bytes(data.tobytes())
        print('Wrote', path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == '__main__':
    main()
