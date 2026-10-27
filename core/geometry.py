"""Pure geometric helpers: easy to unit-test without camera or AI libraries."""
from math import hypot


def distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def normalize_to_roi(x, y, margin=0.15):
    width = 1 - 2 * margin
    if width <= 0:
        raise ValueError("margin must be less than 0.5")
    return (clamp((x - margin) / width), clamp((y - margin) / width))


def pinch_ratio(hand, finger_tip=8):
    """Thumb-to-finger-tip distance relative to palm width; smaller = pinched."""
    if hand is None or len(hand.landmarks) < 21:
        return None
    lm = hand.landmarks
    palm_width = max(distance(lm[5], lm[17]), 0.00001)
    return distance(lm[4], lm[finger_tip]) / palm_width


def is_open_palm(hand):
    """Fallback using finger extension, independent of gesture label confidence."""
    if hand is None or len(hand.landmarks) < 21:
        return False
    lm = hand.landmarks
    palm_width = distance(lm[5], lm[17])
    if palm_width < 0.02:
        return False
    extended = 0
    for base, joint, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b, c = lm[base], lm[joint], lm[tip]
        ux, uy = b.x - a.x, b.y - a.y
        vx, vy = c.x - b.x, c.y - b.y
        lengths = hypot(ux, uy) * hypot(vx, vy)
        if (lengths > 0 and (ux * vx + uy * vy) / lengths > 0.7
                and distance(c, lm[0]) > distance(b, lm[0]) * 1.15):
            extended += 1
    return extended >= 3 and distance(lm[4], lm[5]) > palm_width * 0.35
