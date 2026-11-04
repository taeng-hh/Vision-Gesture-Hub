"""Shared look for every screen: the Sign Galaga palette, type scale and helpers.

Colours are hex strings for Pillow; use bgr() when drawing with OpenCV.
"""
from functools import lru_cache
import numpy as np
from PIL import Image
from ui.ksl_overlay import get_font

# Canvas and window shared by the translator and the game.
WIDTH, HEIGHT = 1280, 860
WINDOW_TITLE = 'Sign Translator'

# Palette (taken from the Galaga screen).
BACKGROUND = '#080c20'
CARD = '#121b32'
CARD_BORDER = '#2b4264'
CARD_DEEP = '#030715'      # Play field / camera well.
CARD_RAISED = '#142139'    # Pop-up cards and chips.
TEXT = '#edf3ff'
MUTED = '#9aaaca'
ACCENT = '#65edc2'         # Mint.
DANGER = '#ff778d'
WARNING = '#ffd166'
RECORDING = '#ff5577'
SKY = '#65edff'
TRACK = '#22314f'          # Empty part of progress bars.

# Type scale.
TITLE = 30
CAPTION = 40
HEADING = 21
BODY = 17
SMALL = 13

RADIUS = 14


def bgr(color):
    """'#rrggbb' -> OpenCV (b, g, r)."""
    value = color.lstrip('#')
    r, g, b = (int(value[i:i+2], 16) for i in (0, 2, 4))
    return b, g, r


def rgb(color):
    value = color.lstrip('#')
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


@lru_cache(maxsize=24)
def font(size):
    return get_font().font_variant(size=size)


def card(draw, box, fill=CARD, outline=CARD_BORDER, radius=RADIUS, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_lines(draw, value, width, size):
    """Character-level wrap (Korean has no spaces to break on reliably)."""
    lines, line = [], ''
    for char in str(value):
        if draw.textlength(line+char, font=font(size)) > width:
            lines.append(line)
            line = ''
        line += char
    lines.append(line)
    return lines


def wrapped_text(draw, value, x, y, width, size=BODY, color=MUTED, max_lines=2, line_gap=7):
    """Draw wrapped text; returns the y below the last drawn line."""
    lines = wrap_lines(draw, value, width, size)[:max_lines]
    for i, line in enumerate(lines):
        draw.text((x, y+i*(size+line_gap)), line, font=font(size), fill=color)
    return y + len(lines)*(size+line_gap)


def fit_size(draw, value, width, sizes):
    """Largest size in `sizes` whose single line fits `width` (else the last)."""
    for size in sizes:
        if draw.textlength(str(value), font=font(size)) <= width:
            return size
    return sizes[-1]


def to_pil(frame_bgr):
    return Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))


def to_bgr(image):
    return np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
