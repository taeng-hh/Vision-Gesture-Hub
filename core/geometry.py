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


def pinch_ratio(hand):
    """Thumb-tip/index-tip distance relative to palm width; smaller = pinched."""
    if hand is None or len(hand.landmarks) < 21:
        return None
    lm = hand.landmarks
    palm_width = max(distance(lm[5], lm[17]), 0.00001)
    return distance(lm[4], lm[8]) / palm_width
