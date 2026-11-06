"""Convert labeled isolated-word videos/segments to the collector's format.

CSV columns: video,label,signer,start_s,end_s. Paths are relative to the CSV.
start_s/end_s are optional; blank end_s means end of video.
This does not parse AI Hub JSON automatically or infer word boundaries.
"""
import argparse
import csv
import json
import uuid
from pathlib import Path
import cv2
from core.detector import GestureDetector
from core.ksl import hand_frame, encode

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/ksl_clips")
    parser.add_argument("--already-mirrored", action="store_true",
                        help="Skip flipping if videos already match webcam mirror view")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if not row.get("label") or not row.get("signer"):
            raise ValueError("Every row needs label and signer")
        video = (args.manifest.resolve().parent / row["video"]).resolve()
        start_s = float(row.get("start_s") or 0)
        end_s = float(row.get("end_s") or "inf")
        if start_s < 0 or end_s <= start_s:
            raise ValueError(f"Invalid time interval: {video}")
        cap = cv2.VideoCapture(str(video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not 1 <= fps <= 240:
                raise ValueError(f"Cannot read video FPS: {video}")
            frames, timestamps = [], []
            index = 0
            # New tracker for each clip, never carry state across signers/videos.
            with GestureDetector(ROOT / "models/gesture_recognizer.task") as detector:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    now = index / fps
                    index += 1
                    if now < start_s:
                        continue
                    if now >= end_s:
                        break
                    if not args.already_mirrored:
                        frame = cv2.flip(frame, 1)
                    frames.append(hand_frame(detector.detect(frame, int(now * 1000))).tolist())
                    timestamps.append(now - start_s)
                    if len(frames) > 1800:
                        raise ValueError("Segment too long; use isolated-word boundaries")
            encode(frames)  # Reject empty/invalid clips before writing.
            sample = {"version": 1, "label": row["label"], "signer": row["signer"],
                      "mirrored": True, "frames": frames, "timestamps_s": timestamps}
            output = args.out / f"{uuid.uuid4().hex}.json"
            output.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
            print("Imported", output.name, row["label"])
        finally:
            cap.release()


if __name__ == "__main__":
    main()
