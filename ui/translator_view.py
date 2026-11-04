"""Demo-ready sign translator screen (same canvas, palette and cards as Galaga).

The camera loop builds a TranslatorState and calls draw_translator(); nothing
here touches recognition or camera state.
"""
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ui import theme
from ui.theme import font

CAMERA_BOX = (28, 116, 858, 616)
RESULT_BOX = (882, 116, 1252, 346)
TRACKING_BOX = (882, 362, 1252, 452)
CONTROLS_BOX = (882, 468, 1252, 616)
DEBUG_BOX = (882, 362, 1252, 616)      # Replaces tracking + controls while open.
SUBTITLE_BOX = (28, 632, 1252, 798)


@dataclass
class TranslatorState:
    camera: np.ndarray                    # Mirrored BGR preview, skeleton already drawn.
    now: float = 0.                       # Drives the blinking REC dot.
    status: str = 'ready'                 # ready | recording | analyzing
    elapsed: float = 0.                   # Seconds recorded so far.
    event: Optional[dict] = None          # Last recognition: accepted/text/score/kind/candidates.
    message: str = ''
    tokens: tuple = ()                    # Recognised word texts, in order.
    sentence: str = ''
    hands: int = 0
    shoulders: bool = False
    keypoints: bool = False
    tracking_note: str = ''
    game_progress: Optional[float] = None  # None: V-sign shortcut off.
    game_note: str = ''
    debug: bool = False
    debug_lines: tuple = ()
    rois: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def _text(draw, x, y, value, size=theme.BODY, color=theme.TEXT, anchor=None):
    draw.text((x, y), str(value), font=font(size), fill=color, anchor=anchor)


def _ellipsis(draw, value, width, size):
    value = str(value)
    if draw.textlength(value, font=font(size)) <= width:
        return value
    while value and draw.textlength(value+'…', font=font(size)) > width:
        value = value[:-1]
    return value+'…'


def _bar(draw, box, fraction, color):
    left, top, right, bottom = box
    radius = (bottom-top)//2
    draw.rounded_rectangle(box, radius=radius, fill=theme.TRACK)
    fill = int((right-left)*max(0., min(1., fraction)))
    if fill >= 2*radius:
        draw.rounded_rectangle((left, top, left+fill, bottom), radius=radius, fill=color)


def _dot(draw, x, y, color, r=6):
    draw.ellipse((x-r, y-r, x+r, y+r), fill=color)


def _key(draw, x, y, label):
    """Key cap; returns its right edge."""
    width = draw.textlength(label, font=font(theme.SMALL)) + 16
    draw.rounded_rectangle((x, y, x+width, y+22), radius=6, fill=theme.CARD_RAISED, outline=theme.CARD_BORDER)
    _text(draw, x+8, y+3, label, theme.SMALL, theme.TEXT)
    return x + width


def _header(draw, state):
    _text(draw, 32, 22, 'SIGN TRANSLATOR', theme.TITLE)
    _text(draw, 34, 66, '실시간 한국수어 번역', 16, theme.MUTED)
    if state.status == 'recording':
        color, label = theme.RECORDING, f'입력 중 {state.elapsed:.1f}초'
        blink = int(state.now*2) % 2 == 0
    elif state.status == 'analyzing':
        color, label, blink = theme.WARNING, '분석 중', True
    else:
        color, label, blink = theme.ACCENT, '입력 대기', True
    width = draw.textlength(label, font=font(20)) + 64
    left = 1252 - width
    theme.card(draw, (left, 30, 1252, 84), theme.CARD, color, radius=27, width=2)
    _dot(draw, left+28, 57, color if blink else theme.TRACK, 7)
    _text(draw, left+46, 44, label, 20, color)


def _camera(image, draw, state):
    outline = theme.RECORDING if state.status == 'recording' else theme.CARD_BORDER
    theme.card(draw, CAMERA_BOX, theme.CARD_DEEP, outline, width=2 if state.status == 'recording' else 1)
    _text(draw, 50, 130, 'LIVE CAMERA', theme.SMALL, theme.MUTED)
    left, top, width, height = 48, 156, 790, 444
    frame = state.camera
    if frame is None or frame.size == 0:
        _text(draw, left+width//2, top+height//2, '카메라 영상 없음', theme.BODY, theme.MUTED, 'mm')
        return
    h, w = frame.shape[:2]
    scale = min(width/w, height/h)
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    image.paste(theme.to_pil(resized), (left+(width-size[0])//2, top+(height-size[1])//2))


def _result(draw, state):
    theme.card(draw, RESULT_BOX)
    x, right = 902, 1232
    _text(draw, x, 134, '인식 결과', theme.SMALL, theme.MUTED)
    event = state.event
    if event and event.get('kind') in ('word', 'sentence') and event.get('accepted'):
        kind = '문장' if event['kind'] == 'sentence' else '단어'
        width = draw.textlength(kind, font=font(12)) + 16
        draw.rounded_rectangle((right-width, 130, right, 152), radius=11, outline=theme.ACCENT)
        _text(draw, right-width/2, 141, kind, 12, theme.ACCENT, 'mm')
    if event is None:
        _text(draw, x, 176, '아직 인식한 동작이 없어요', 22, theme.MUTED)
        _bar(draw, (x, 236, right-74, 244), 0, theme.ACCENT)
    elif event.get('accepted'):
        text = event.get('text') or ''
        size = theme.fit_size(draw, text, right-x, (44, 36, 30, 26))
        bottom = theme.wrapped_text(draw, text, x, 168, right-x, size, theme.ACCENT, 2, 6)
        score = float(event.get('score') or 0)
        bar_y = max(bottom+10, 236)
        _bar(draw, (x, bar_y, right-74, bar_y+8), score, theme.ACCENT)
        _text(draw, right, bar_y+4, f'{score:.0%}', 15, theme.TEXT, 'rm')
    else:
        _text(draw, x, 170, '다시 해주세요', 34, theme.WARNING)
        candidates = [f'{t} {s:.0%}' for t, s in event.get('candidates', ())[:3] if t]
        if candidates:
            _text(draw, x, 218, _ellipsis(draw, '후보  ' + ' · '.join(candidates), right-x, 15), 15, theme.MUTED)
        score = float(event.get('score') or 0)
        _bar(draw, (x, 248, right-74, 256), score, theme.WARNING)
        _text(draw, right, 252, f'{score:.0%}', 15, theme.MUTED, 'rm')
    if state.message:
        theme.wrapped_text(draw, state.message, x, 284, right-x, 14, theme.MUTED, 2, 6)


def _tracking(draw, state):
    theme.card(draw, TRACKING_BOX)
    _text(draw, 902, 376, '추적 상태', theme.SMALL, theme.MUTED)
    items = ((f'손 {state.hands}/2', state.hands > 0), ('양어깨', state.shoulders), ('키포인트', state.keypoints))
    x = 902
    for label, ok in items:
        _dot(draw, x+6, 412, theme.ACCENT if ok else theme.DANGER)
        _text(draw, x+18, 402, label, 16, theme.TEXT if ok else theme.MUTED)
        x += 18 + draw.textlength(label, font=font(16)) + 26
    if state.tracking_note:
        _text(draw, 902, 428, _ellipsis(draw, state.tracking_note, 330, theme.SMALL), theme.SMALL, theme.MUTED)


def _controls(draw, state):
    theme.card(draw, CONTROLS_BOX)
    _text(draw, 902, 482, '조작', theme.SMALL, theme.MUTED)
    rows = (('Space', '입력 시작 / 완료'), ('Backspace', '마지막 단어 지우기'), ('C', '초기화'))
    for i, (key, label) in enumerate(rows):
        y = 504 + i*27
        _key(draw, 902, y, key)
        _text(draw, 1010, y+2, label, 15, theme.TEXT)
    y = 504 + 3*27
    _key(draw, 902, y, 'V 손 1.2초')
    if state.game_progress is None:
        _text(draw, 1010, y+2, _ellipsis(draw, state.game_note or '게임 전환 꺼짐', 222, 15), 15, theme.MUTED)
    else:
        recording = state.status == 'recording'
        _text(draw, 1010, y+2, '녹화 중엔 꺼짐' if recording else '게임 열기', 15, theme.MUTED if recording else theme.TEXT)
        _bar(draw, (1120, y+8, 1232, y+16), 0 if recording else state.game_progress, theme.ACCENT)


def _debug(image, draw, state):
    theme.card(draw, DEBUG_BOX, theme.CARD, theme.WARNING)
    _text(draw, 902, 376, '디버그', theme.SMALL, theme.WARNING)
    _text(draw, 1232, 376, 'D로 닫기', theme.SMALL, theme.MUTED, 'ra')
    y = 398
    for line in state.debug_lines[:6]:
        _text(draw, 902, y, _ellipsis(draw, line, 330, theme.SMALL), theme.SMALL, theme.TEXT)
        y += 18
    rois = state.rois
    if rois is not None and np.asarray(rois).shape == (3, 224, 224, 3):
        for i, (roi, label) in enumerate(zip(rois, ('왼손', '오른손', '상체'))):
            left = 902 + i*(80+10)
            _text(draw, left, 512, label, 11, theme.MUTED)
            image.paste(Image.fromarray(np.ascontiguousarray(roi)).resize((80, 80)), (left, 528))
    else:
        _text(draw, 902, 528, 'RGB 입력 ROI 없음 (CNN 미실행)', 12, theme.MUTED)


def _subtitle(draw, state):
    theme.card(draw, SUBTITLE_BOX)
    _text(draw, 50, 644, '자막', theme.SMALL, theme.MUTED)
    x, y, limit = 50, 664, 1230
    shown = 0
    for token in state.tokens:
        width = draw.textlength(token, font=font(16)) + 28
        if x + width > limit - 60:
            break
        draw.rounded_rectangle((x, y, x+width, y+30), radius=15, fill=theme.CARD_RAISED, outline=theme.ACCENT)
        _text(draw, x+width/2, y+15, token, 16, theme.TEXT, 'mm')
        x += width + 8
        shown += 1
    if shown < len(state.tokens):
        _text(draw, x+4, y+5, f'+{len(state.tokens)-shown}', 16, theme.MUTED)
    width = limit - 50
    if state.sentence:
        size = theme.fit_size(draw, state.sentence, width, (theme.CAPTION, 34, 28))
        theme.wrapped_text(draw, state.sentence, 50, 708, width, size, theme.TEXT, 2, 6)
    else:
        _text(draw, 50, 718, '인식한 단어와 문장이 여기에 자막으로 표시됩니다', 22, theme.MUTED)


def draw_translator(state):
    """TranslatorState -> 1280x860 BGR frame."""
    image = Image.new('RGB', (theme.WIDTH, theme.HEIGHT), theme.BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, state)
    _camera(image, draw, state)
    _result(draw, state)
    if state.debug:
        _debug(image, draw, state)
    else:
        _tracking(draw, state)
        _controls(draw, state)
    _subtitle(draw, state)
    _text(draw, 35, 818, 'Space 입력    Backspace 지우기    C 초기화    D 디버그    Q / ESC 종료', 14, theme.MUTED)
    _text(draw, 1245, 818, '판정 모드  1 키포인트 · 2 영상 · 3 CNN 보조', 14, theme.MUTED, 'ra')
    return theme.to_bgr(image)
