"""Download official MediaPipe Pose Landmarker Lite; preserve Korean paths."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
       'pose_landmarker_lite/float16/1/pose_landmarker_lite.task')


def main():
    target = ROOT / 'models/pose_landmarker_lite.task'
    target.parent.mkdir(exist_ok=True)
    if target.exists() and target.stat().st_size > 100000:
        print('Pose model already exists:', target)
        return
    temp = target.with_suffix('.download')
    try:
        with urlopen(URL, timeout=60) as response, temp.open('wb') as file:
            while chunk := response.read(1024 * 1024): file.write(chunk)
        if temp.stat().st_size < 100000: raise RuntimeError('Model download too small')
        temp.replace(target)
        print('Downloaded pose model:', target)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == '__main__': main()
