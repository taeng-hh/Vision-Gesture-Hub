"""Entry point: camera -> shared detector -> one active mode -> optional OS action."""
import argparse
import time
from pathlib import Path

import cv2
import pyautogui

from core.detector import GestureDetector
from core.mode_manager import KEY_MODES, Mode, ModeManager
from modes.pc_control import PCController
from modes.presentation import PresentationController
from modes.sign_language import SignLanguageMode
from ui.overlay import draw_overlay

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Vision Gesture Hub starter")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models" / "gesture_recognizer.task")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    manager = ModeManager()
    pc = PCController()
    presentation = PresentationController()
    sign = SignLanguageMode()
    start = time.monotonic()

    try:
        with GestureDetector(args.model, max_hands=2) as detector:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Camera frame read failed")
                    break
                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                observation = detector.detect(frame, int((now-start)*1000))
                primary = observation.primary
                status = (f"Hand gesture: {primary.gesture} ({primary.score:.2f})"
                          if primary else "No hand detected")

                try:
                    if manager.mode == Mode.SIGN:
                        status = sign.update(observation)
                    elif manager.mode == Mode.PC and manager.actions_enabled:
                        status = pc.update(primary, now)
                    elif manager.mode == Mode.PRESENTATION and manager.actions_enabled:
                        status = presentation.update(primary, now)
                except pyautogui.FailSafeException:
                    # Moving pointer to a screen corner is an emergency stop.
                    manager.actions_enabled = False
                    pc.reset()
                    presentation.reset()
                    status = "PyAutoGUI fail-safe: actions disabled"

                draw_overlay(frame, observation, manager, status)
                cv2.imshow("Vision Gesture Hub", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in KEY_MODES:
                    manager.select(KEY_MODES[key])
                    pc.reset()
                    presentation.reset()
                    print("Mode:", manager.mode.value, "(actions OFF)")
                elif key == ord("e"):
                    enabled = manager.toggle_actions()
                    pc.reset()
                    presentation.reset()
                    print("OS actions:", "ON" if enabled else "OFF")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
