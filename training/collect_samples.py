"""Collect per-frame landmark CSV examples for a future custom sign classifier.

Example: python -m training.collect_samples --label HELLO --count 50
Stores landmarks only. This is NOT a continuous sign-language dataset.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
from core.detector import GestureDetector
from ui.overlay import draw_overlay
from core.mode_manager import ModeManager

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ["label", "timestamp_ms"] + [
    f"h{hand}_{axis}{idx}" for hand in range(2)
    for idx in range(21) for axis in ("x", "y", "z")
]


def flatten_hand(hand):
    if hand is None:
        return [""] * 63
    return [value for point in hand.landmarks for value in (point.x, point.y, point.z)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="One class name per run")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "samples.csv")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or args.out.stat().st_size == 0
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot access camera")
    start = time.monotonic()
    stored = 0
    print("Press SPACE to start capturing landmarks. Q to quit.")
    capturing = False
    try:
        with GestureDetector(ROOT / "models" / "gesture_recognizer.task") as detector, \
                open(args.out, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(HEADERS)
            while stored < args.count:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                timestamp = int((time.monotonic() - start) * 1000)
                obs = detector.detect(frame, timestamp)
                # Avoid treating no-hand frames as valid sign examples.
                if capturing and obs.hands:
                    row = [args.label, timestamp]
                    row += flatten_hand(obs.hands[0])
                    row += flatten_hand(obs.hands[1] if len(obs.hands) > 1 else None)
                    writer.writerow(row)
                    stored += 1
                draw_overlay(frame, obs, ModeManager(),
                             f"Label={args.label} {stored}/{args.count} | SPACE: record | Q: quit")
                cv2.imshow("Collect landmark samples", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    capturing = not capturing
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Recorded {stored} frames to {args.out}")


if __name__ == "__main__":
    main()
