"""Debug UI in OpenCV (English-only text because cv2.putText is ASCII-centric)."""
import cv2


def draw_overlay(frame, observation, manager, status):
    height, width = frame.shape[:2]
    for hand in observation.hands:
        for p in hand.landmarks:
            x = int(p.x * width)
            y = int(p.y * height)
            cv2.circle(frame, (x, y), 3, (0, 220, 80), -1)

    active = "ON" if manager.actions_enabled else "OFF"
    lines = [
        f"Mode: {manager.mode.value} | OS actions: {active}",
        "0: Idle 1: Sign 2: PC 3: Slides",
        "E: Enable/disable actions | Q: Quit",
        f"{status}",
    ]
    for line_no, line in enumerate(lines):
        y = 25 + 30 * line_no
        cv2.rectangle(frame, (5, y - 20), (min(width-5, 620), y + 5), (25, 25, 25), -1)
        cv2.putText(frame, line[:78], (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
