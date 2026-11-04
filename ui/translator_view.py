"""Demo-ready sign translator screen (same canvas, palette and cards as Galaga).

The camera loop builds a TranslatorState and calls draw_translator(); nothing
here touches recognition or camera state.
"""
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ui import theme
from ui.theme import font

CAMERA_BOX = (28, 116, 858, 616)
CAMERA_VIEW = (48, 156, 838, 600)      # 790x444 (16:9) image well inside the camera card.
RESULT_BOX = (882, 116, 1252, 346)
TRACKING_BOX = (882, 362, 1252, 452)
CONTROLS_BOX = (882, 468, 1252, 616)
DEBUG_BOX = (882, 362, 1252, 616)      # Replaces tracking + controls while open.
SUBTITLE_BOX = (28, 632, 1252, 798)
FLASH_SECONDS = .6                     # New-result highlight.
FRESH_SECONDS = 2.                     # Newest token chip stays filled.


@dataclass
class TranslatorState:
    camera: Optional[np.ndarray]          # Mirrored BGR preview, skeleton already drawn.
    now: float = 0.                       # Drives blinking / animated dots.
    status: str = 'ready'                 # ready | recording | analyzing | reconnecting
    elapsed: float = 0.                   # Seconds recorded so far.
    record_limit: float = 12.             # Recording auto-cancels after this.
    event: Optional[dict] = None          # Last recognition: accepted/text/score/kind/candidates.
    event_age: Optional[float] = None     # Seconds since `event` was newly accepted.
    message: str = ''
    tokens: tuple = ()                    # Recognised word texts, in order.
    sentence: str = ''
    history: tuple = ()                   # Earlier sentences, oldest first.
    recent: tuple = ()                    # Latest accepted results [(text, score)], newest first.
    hands: int = 0
    shoulders: bool = False
    keypoints: bool = False
    tracking_note: str = ''
    camera_note: str = ''                 # Banner over the camera (e.g. reconnecting).
    game_progress: Optional[float] = None  # None: V-sign shortcut off.
    game_note: str = ''
    overlay: Optional[tuple] = None       # (title, text) centred pop-up, e.g. game transition.
    debug: bool = False
    debug_lines: tuple = ()
    rois: Optional[np.ndarray] = None


def _text(draw, x, y, value, size=theme.BODY, color=theme.TEXT, anchor=None, face=None):
    draw.text((x, y), str(value), font=face or font(size), fill=color, anchor=anchor)


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


def _pill(draw, x, y, label, size, filled=False, color=theme.ACCENT, height=30):
    """Rounded chip; returns its right edge."""
    width = draw.textlength(label, font=font(size)) + 2*max(10, height//2-1)
    draw.rounded_rectangle((x, y, x+width, y+height), radius=height//2,
                           fill=color if filled else theme.CARD_RAISED, outline=color)
    _text(draw, x+width/2, y+height/2, label, size, theme.BACKGROUND if filled else theme.TEXT, 'mm')
    return x + width


def _translucent(image, box, color=theme.BACKGROUND, alpha=190, radius=10):
    """Darken `box` in place (only that region is composited)."""
    left, top, right, bottom = (int(v) for v in box)
    region = image.crop((left, top, right, bottom)).convert('RGBA')
    layer = Image.new('RGBA', region.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, right-left-1, bottom-top-1), radius=radius,
                                            fill=(*theme.rgb(color), alpha))
    image.paste(Image.alpha_composite(region, layer).convert('RGB'), (left, top))


def _hand_icon(draw, x, y, color=theme.ACCENT):
    """Small open-hand mark drawn with shapes (no emoji/font dependency)."""
    draw.rounded_rectangle((x+5, y+18, x+29, y+40), radius=8, fill=color)          # Palm.
    for i, height in enumerate((17, 22, 20, 14)):                                  # Fingers.
        left = x+6+i*6
        draw.rounded_rectangle((left, y+22-height, left+4, y+24), radius=2, fill=color)
    draw.rounded_rectangle((x-1, y+21, x+9, y+27), radius=3, fill=color)          # Thumb.


def _header(draw, state):
    _hand_icon(draw, 34, 24)
    _text(draw, 76, 20, 'SIGN TRANSLATOR', theme.TITLE, face=theme.bold(theme.TITLE))
    _text(draw, 78, 64, '실시간 한국수어 번역', 16, theme.MUTED)
    if state.status == 'recording':
        color, label = theme.RECORDING, f'입력 중 {state.elapsed:.1f}초'
        blink = int(state.now*2) % 2 == 0
    elif state.status == 'analyzing':
        color, label, blink = theme.WARNING, '분석 중' + '.'*(int(state.now*3) % 3 + 1), True
    elif state.status == 'reconnecting':
        color, label, blink = theme.WARNING, '카메라 연결 중', int(state.now*2) % 2 == 0
    else:
        color, label, blink = theme.ACCENT, '입력 대기', True
    # Fixed width per status so animated dots do not make the badge jitter.
    sizing = {'analyzing': '분석 중...', 'recording': '입력 중 00.0초'}.get(state.status, label)
    width = draw.textlength(sizing, font=font(20)) + 64
    left = 1252 - width
    theme.card(draw, (left, 30, 1252, 84), theme.CARD, color, radius=27, width=2)
    _dot(draw, left+28, 57, color if blink else theme.TRACK, 7)
    _text(draw, left+46, 44, label, 20, color)


def _brackets(draw, box, color, length=26, width=3):
    left, top, right, bottom = box
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
        draw.line((x, y, x+dx*length, y), fill=color, width=width)
        draw.line((x, y, x, y+dy*length), fill=color, width=width)


def _camera(image, draw, state):
    recording = state.status == 'recording'
    theme.card(draw, CAMERA_BOX, theme.CARD_DEEP, theme.RECORDING if recording else theme.CARD_BORDER,
               width=2 if recording else 1)
    live = state.status != 'reconnecting'
    _dot(draw, 54, 137, theme.RECORDING if live and (not recording or int(state.now*2) % 2 == 0) else theme.TRACK, 5)
    _text(draw, 66, 130, 'LIVE CAMERA', theme.SMALL, theme.MUTED)
    left, top, right, bottom = CAMERA_VIEW
    width, height = right-left, bottom-top
    frame = state.camera
    if frame is None or frame.size == 0:
        _text(draw, left+width//2, top+height//2, '카메라 영상 없음', theme.BODY, theme.MUTED, 'mm')
    else:
        h, w = frame.shape[:2]
        scale = min(width/w, height/h)
        size = (max(1, round(w*scale)), max(1, round(h*scale)))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        image.paste(theme.to_pil(resized), (left+(width-size[0])//2, top+(height-size[1])//2))
    _brackets(draw, (left+10, top+10, right-10, bottom-10), theme.RECORDING if recording else theme.ACCENT)
    note = state.camera_note or ('' if state.keypoints or state.status == 'analyzing' else state.tracking_note)
    if note:
        banner = (left+150, bottom-62, right-150, bottom-20)
        _translucent(image, banner)
        draw = ImageDraw.Draw(image)
        _dot(draw, banner[0]+22, (banner[1]+banner[3])//2, theme.WARNING, 6)
        _text(draw, banner[0]+38, (banner[1]+banner[3])//2, _ellipsis(draw, note, banner[2]-banner[0]-56, 17),
              17, theme.TEXT, 'lm')
    if recording:
        _bar(draw, (left, 606, right, 611), state.elapsed/max(state.record_limit, 1e-6), theme.RECORDING)
    return draw


def _result(draw, state):
    age = state.event_age
    flashing = age is not None and 0 <= age < FLASH_SECONDS
    theme.card(draw, RESULT_BOX, theme.CARD, theme.ACCENT if flashing else theme.CARD_BORDER, width=3 if flashing else 1)
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
        if flashing:  # Pop in: start slightly larger and settle.
            size = round(size*(1+.18*(1-age/FLASH_SECONDS)))
            size = min(size, theme.fit_size(draw, text, right-x, tuple(range(size, 25, -2))))
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
        if state.recent:   # One line; the recent chips sit just below.
            _text(draw, x, 272, _ellipsis(draw, state.message, right-x, 14), 14, theme.MUTED)
        else:
            lines = theme.wrap_lines(draw, state.message, right-x, 14)
            if len(lines) > 2:
                lines = [lines[0], _ellipsis(draw, ''.join(lines[1:]), right-x, 14)]
            for i, line in enumerate(lines):
                _text(draw, x, 272+i*21, line, 14, theme.MUTED)
    if state.recent:
        _text(draw, x, 312, '최근', 12, theme.MUTED)
        cx = x + 34
        for text, score in state.recent[:3]:
            label = f'{text} {score:.0%}'
            width = draw.textlength(label, font=font(12)) + 20
            if cx + width > right:
                break
            cx = _pill(draw, cx, 306, label, 12, height=24, color=theme.CARD_BORDER) + 6


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
    rows = (('Space', '입력 시작 / 완료'), ('Backspace', '마지막 단어 지우기'), ('C', '다음 문장 / 초기화'))
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
    left, right = 50, 1230
    if state.history:
        earlier = '   ·   '.join(state.history[-2:])
        _text(draw, right, 644, _ellipsis(draw, '이전  ' + earlier, 900, 14), 14, theme.MUTED, 'ra')
    top = 666
    if state.tokens:
        x, shown = left, 0
        fresh = state.event_age is not None and state.event_age < FRESH_SECONDS
        for i, token in enumerate(state.tokens):
            width = draw.textlength(token, font=font(16)) + 28
            if x + width > right - 60:
                break
            x = _pill(draw, x, top, token, 16, filled=fresh and i == len(state.tokens)-1) + 8
            shown += 1
        if shown < len(state.tokens):
            _text(draw, x+4, top+5, f'+{len(state.tokens)-shown}', 16, theme.MUTED)
        top += 38
    area = SUBTITLE_BOX[3] - 12 - top   # Vertical space left for the sentence.
    width = right - left
    if state.sentence:
        size = theme.fit_size(draw, state.sentence, width, (48, 42, 36, 30))
        lines = theme.wrap_lines(draw, state.sentence, width, size)[:2]
        if len(lines) > 1 and size > 30:
            size = 30
            lines = theme.wrap_lines(draw, state.sentence, width, size)[:2]
        block = len(lines)*(size+6) - 6
        y = top + max(0, (area-block)//2) - 4
        for i, line in enumerate(lines):
            _text(draw, left, y+i*(size+6), line, size, theme.TEXT, face=theme.bold(size))
    else:
        _text(draw, left, top + area//2, '인식한 단어와 문장이 여기에 자막으로 표시됩니다', 22, theme.MUTED, 'lm')


def _overlay(image, state):
    title, text = state.overlay
    _translucent(image, (0, 0, theme.WIDTH, theme.HEIGHT), alpha=150, radius=0)
    draw = ImageDraw.Draw(image)
    theme.card(draw, (315, 340, 965, 500), theme.CARD_RAISED, theme.ACCENT, width=2)
    _text(draw, 640, 392, title, 30, theme.ACCENT, 'mm', face=theme.bold(30))
    _text(draw, 640, 450, text, 17, theme.MUTED, 'mm')


def draw_translator(state):
    """TranslatorState -> 1280x860 BGR frame."""
    image = Image.new('RGB', (theme.WIDTH, theme.HEIGHT), theme.BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, state)
    draw = _camera(image, draw, state)
    _result(draw, state)
    if state.debug:
        _debug(image, draw, state)
    else:
        _tracking(draw, state)
        _controls(draw, state)
    _subtitle(draw, state)
    _text(draw, 35, 818, 'Space 입력    Backspace 지우기    C 다음 문장    D 디버그    Q / ESC 종료', 14, theme.MUTED)
    _text(draw, 1245, 818, '판정 모드  1 키포인트 · 2 영상 · 3 CNN 보조', 14, theme.MUTED, 'ra')
    if state.overlay:
        _overlay(image, state)
    return theme.to_bgr(image)
