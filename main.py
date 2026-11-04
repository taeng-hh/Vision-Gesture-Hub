"""Entry point.

python main.py               sign translator (keypoint GRU + CNN assist, fusion/)
python main.py --legacy-ksl  previous keypoint-only GRU screen
python main.py --runner      Sign Galaga directly
python main.py --hub         old mode menu
"""
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import sys
    if "--runner" in sys.argv:
        sys.argv.remove("--runner")
        import runpy
        return runpy.run_module("modes.mini_game", run_name="__main__")
    if "--legacy-ksl" in sys.argv:
        # Previous keypoint-only GRU screen (sign_language_webcam.py).
        sys.argv.remove("--legacy-ksl")
        from sign_language_webcam import main as run_sign_language_webcam
        return run_sign_language_webcam()
    if "--hub" not in sys.argv:
        # Default: keypoint-first sign translator with CNN assistance and the
        # V hand shape -> Galaga shortcut. --live-ksl is kept as an old alias.
        if "--live-ksl" in sys.argv:
            sys.argv.remove("--live-ksl")
        from fusion.cli import main as run_sign_translator
        return run_sign_translator()
    sys.argv.remove("--hub")
    # Load desktop-control dependencies only for the optional legacy hub.
    import cv2
    import pyautogui
    from core.detector import GestureDetector
    from core.mode_manager import KEY_MODES, Mode, ModeManager
    from core.navigation import GestureNavigation
    from modes.mini_game import run_game
    from modes.presentation import PresentationController
    from modes.sign_language_placeholder import SignLanguageMode
    from ui.overlay import draw_overlay

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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; camera backend may ignore it.

    manager = ModeManager()
    navigation = GestureNavigation()
    presentation = PresentationController()
    sign = SignLanguageMode()
    start = time.monotonic()

    try:
        cv2.namedWindow("Vision Gesture Hub", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision Gesture Hub", 1280, 960)
        with GestureDetector(args.model, max_hands=2) as detector:
            while True:
                if manager.mode == Mode.PC:
                    cap.release()
                    cv2.destroyWindow("Vision Gesture Hub")
                    try:
                        run_game(args.camera)
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(f"Galaga: {exc}")
                    finally:
                        manager.select(Mode.IDLE)
                        navigation.reset()
                        cap.open(args.camera)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cv2.namedWindow("Vision Gesture Hub", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Vision Gesture Hub", 1280, 960)
                    continue
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

                selection = navigation.update(primary, manager.mode, now)
                if selection is not None:
                    if selection == "TOGGLE":
                        manager.toggle_actions()
                    else:
                        manager.select(Mode.IDLE if selection == "HOME" else Mode(selection))
                    presentation.reset()
                # Menu interaction takes priority over mouse/keyboard actions.
                navigating = navigation.hover is not None or selection is not None
                if navigating:
                    presentation.reset()

                try:
                    if manager.mode == Mode.IDLE:
                        status = f"Point at a card and hold for {navigation.dwell_s:.1f} seconds"
                    elif navigating:
                        status = "Hold to select; move away to resume"
                    elif manager.mode == Mode.SIGN:
                        status = sign.update(observation)
                    elif manager.mode == Mode.PRESENTATION and manager.actions_enabled:
                        status = presentation.update(primary, now)
                except pyautogui.FailSafeException:
                    # Moving pointer to a screen corner is an emergency stop.
                    manager.actions_enabled = False
                    presentation.reset()
                    status = "PyAutoGUI fail-safe: actions disabled"

                draw_overlay(frame, observation, manager, status, navigation)
                cv2.imshow("Vision Gesture Hub", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in KEY_MODES:
                    manager.select(KEY_MODES[key])
                    navigation.reset()
                    presentation.reset()
                    print("Mode:", manager.mode.value, "(actions OFF)")
                elif key == ord("e"):
                    enabled = manager.toggle_actions()
                    navigation.reset()
                    presentation.reset()
                    print("OS actions:", "ON" if enabled else "OFF")
    finally:
        presentation.reset()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
