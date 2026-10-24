"""Download the official MediaPipe Gesture Recognizer bundle (not committed to Git)."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "gesture_recognizer.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)


def main():
    TARGET.parent.mkdir(exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 100_000:
        print(f"Model already exists: {TARGET}")
        return
    print(f"Downloading official MediaPipe model to {TARGET}...")
    temp = TARGET.with_suffix(".download")
    try:
        with urlopen(URL, timeout=60) as response, open(temp, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if temp.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temp.replace(TARGET)
        print(f"Saved {TARGET} ({TARGET.stat().st_size:,} bytes)")
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
