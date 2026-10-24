"""The only module that needs to know MediaPipe's result format."""
from pathlib import Path
import cv2
import mediapipe as mp
from core.types import Hand, Landmark, Observation


class GestureDetector:
    def __init__(self, model_path: Path, max_hands: int = 2):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run: python scripts/download_model.py"
            )
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
        self._last_timestamp = -1

    def detect(self, frame_bgr, timestamp_ms: int) -> Observation:
        # MediaPipe VIDEO timestamps must always increase.
        timestamp_ms = max(timestamp_ms, self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands = []
        for index, raw_landmarks in enumerate(result.hand_landmarks):
            category = (
                result.gestures[index][0]
                if index < len(result.gestures) and result.gestures[index]
                else None
            )
            handed = (
                result.handedness[index][0].category_name
                if index < len(result.handedness) and result.handedness[index]
                else "Unknown"
            )
            hands.append(Hand(
                landmarks=tuple(Landmark(p.x, p.y, p.z) for p in raw_landmarks),
                gesture=category.category_name if category else "None",
                score=float(category.score) if category else 0.0,
                handedness=handed,
            ))
        return Observation(tuple(hands))

    def close(self):
        self._recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
