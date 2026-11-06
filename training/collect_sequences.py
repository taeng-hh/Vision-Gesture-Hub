"""Record one complete sign per clip; each SPACE toggles recording."""
import argparse
import json
import time
import uuid
from pathlib import Path
import cv2
from core.detector import GestureDetector
from core.ksl import hand_frame

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--signer", required=True, help="Anonymous consistent signer ID")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data/ksl_clips")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    recording = False
    frames, timestamps = [], []
    start = time.monotonic()
    try:
        with GestureDetector(ROOT / "models/gesture_recognizer.task") as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                observation = detector.detect(frame, int((now - start) * 1000))
                if recording:
                    frames.append(hand_frame(observation).tolist())
                    timestamps.append(now)
                cv2.putText(frame, f"{'REC' if recording else 'READY'} {len(frames)} frames | SPACE: start/save Q: quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 100), 1)
                cv2.imshow("KSL sequence collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if recording:
                        if len(frames) >= 4:
                            sample = {"version": 1, "label": args.label, "signer": args.signer,
                                      "mirrored": True, "frames": frames,
                                      "timestamps_s": [t - timestamps[0] for t in timestamps]}
                            path = args.out / f"{uuid.uuid4().hex}.json"
                            path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
                            print("Saved", path)
                        else:
                            print("Discarded: too short")
                        recording = False
                    else:
                        recording = True
                        frames, timestamps = [], []
                if recording and len(frames) >= 300:
                    print("Discarded: clip exceeded 300 frames")
                    recording = False
                    frames, timestamps = [], []
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
