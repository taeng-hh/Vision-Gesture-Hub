"""Webcam -> hands + shoulders -> word/sentence recognition -> Korean text."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import cv2
import mediapipe as mp
from core.aihub_words import AIHubWordModel
from core.detector import GestureDetector
from core.live_sign import LiveSignSession, MixedLiveSignSession, webcam_frame
from ui.ksl_overlay import draw_ksl_overlay

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/ksl30_robust_v2')
    parser.add_argument('--pose-model', type=Path, default=ROOT / 'models/pose_landmarker_lite.task')
    args = parser.parse_args()
    if not args.pose_model.exists():
        parser.error('Run: .venv\\Scripts\\python.exe scripts\\download_pose_model.py')
    if args.model.is_dir():
        from ksl.predict import Recognizer
        session = MixedLiveSignSession(Recognizer(args.model))
    elif args.model.is_file():
        session = LiveSignSession(AIHubWordModel.load(args.model))
    else:
        parser.error(f'Model not found: {args.model}')
    # Tracker input is NOT mirrored; only the preview is mirrored.
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=args.pose_model.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    cap = None
    try:
        with ExitStack() as stack:
            detector = stack.enter_context(GestureDetector(ROOT / 'models/gesture_recognizer.task'))
            pose = stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened(): raise RuntimeError('Camera unavailable; try --camera 1')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            last_timestamp = -1
            while True:
                ok, raw = cap.read()
                if not ok: break
                now = time.monotonic()
                timestamp = max(last_timestamp + 1, int(now * 1000))
                last_timestamp = timestamp
                observation = detector.detect(raw, timestamp)
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                body = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                landmarks = body.pose_landmarks[0] if body.pose_landmarks else None
                h, w = raw.shape[:2]
                frame = webcam_frame(observation, landmarks, w, h)
                session.update(frame, now)
                display = cv2.flip(raw, 1)
                if frame is not None:
                    for p in frame[:42]:
                        if p[2] >= .2: cv2.circle(display, (w - 1 - int(p[0]), int(p[1])), 3, (0, 230, 90), -1)
                    for i in (44, 47):
                        p = frame[i]
                        cv2.circle(display, (w - 1 - int(p[0]), int(p[1])), 6, (255, 160, 0), -1)
                cv2.putText(display, 'REC' if session.recording else 'READY | SPACE: record | Q: quit',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (0, 0, 255) if session.recording else (0, 255, 100), 2)
                lines = session.display_lines
                if frame is None and not session.recording:
                    lines[2] = '상반신과 어깨, 손을 카메라에 보이게 해 주세요.'
                draw_ksl_overlay(display, lines)
                cv2.imshow('KSL Webcam - Words and Sentences', display)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q'), 27): break
                if cv2.getWindowProperty('KSL Webcam - Words and Sentences', cv2.WND_PROP_VISIBLE) < 1: break
                session.handle_key(key, now)
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__': main()
