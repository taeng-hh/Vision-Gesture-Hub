"""Debug UI in OpenCV (English-only text because cv2.putText is ASCII-centric)."""
import cv2
from core.mode_manager import Mode
from core.navigation import targets_for


def draw_overlay(frame, observation, manager, status, navigation=None):
    height, width = frame.shape[:2]
    for hand in observation.hands:
        for p in hand.landmarks:
            x = int(p.x * width)
            y = int(p.y * height)
            cv2.circle(frame, (x, y), 3, (0, 220, 80), -1)

    active = "ON" if manager.actions_enabled else "OFF"
    lines = [
        f"Mode: {manager.mode.value} | OS actions: {active}",
        "0: Home 1: Sign 2: PC 3: Slides",
        "E: Enable/disable actions | Q: Quit",
        f"{status}",
    ]
    if manager.mode == Mode.PC:
        lines.append("V sign: hold 1s to open ChatGPT (actions ON)")
        lines.append("Thumb + index + pinky: hold 1s to open Paint")
    for line_no, line in enumerate(lines):
        y = (100 if navigation is not None else 25) + 25 * line_no
        cv2.rectangle(frame, (5, y - 20), (min(width-5, 620), y + 5), (25, 25, 25), -1)
        cv2.putText(frame, line[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if navigation is not None:
        if manager.mode == Mode.IDLE:
            cv2.putText(frame, "VISION GESTURE HUB", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, "Point with your index finger. Hold to enter.",
                        (20, height - 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        for target in targets_for(manager.mode):
            x1, y1, x2, y2 = target.bounds
            left, top, right, bottom = (int(x1 * width), int(y1 * height),
                                         int(x2 * width), int(y2 * height))
            hovered = navigation.hover == target
            color = (80, 220, 160) if hovered else (180, 160, 100)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            label = target.label
            if target.key == "TOGGLE":
                label = "DISABLE" if manager.actions_enabled else "ENABLE"
            cv2.putText(frame, label, (left + 8, top + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            if hovered:
                end = left + int((right - left) * navigation.progress)
                cv2.rectangle(frame, (left, bottom - 8), (end, bottom), color, -1)
        if navigation.pointer is not None:
            x, y = navigation.pointer
            cv2.circle(frame, (int(x * width), int(y * height)),
                       10, (0, 240, 255), 2)
    return frame
