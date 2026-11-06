"""Small sequence-template baseline; labels come exclusively from recorded data."""
import json
from pathlib import Path
import numpy as np

VERSION = 1
STEPS = 24


def hand_frame(observation):
    """Fixed Left/Right slots. Unknown or duplicate handedness is dropped."""
    frame = np.zeros((2, 21, 3), dtype=np.float32)
    for slot, side in enumerate(("Left", "Right")):
        hands = [h for h in observation.hands if h.handedness == side]
        if len(hands) == 1 and len(hands[0].landmarks) == 21:
            frame[slot] = [(p.x, p.y, p.z) for p in hands[0].landmarks]
    return frame


def encode(frames):
    raw = np.asarray(frames, dtype=np.float32)
    if raw.ndim != 4 or raw.shape[1:] != (2, 21, 3) or len(raw) < 4:
        raise ValueError("Expected at least 4 frames shaped (T, 2, 21, 3)")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite landmarks")
    present = np.any(raw != 0, axis=(2, 3))
    if not present.any():
        raise ValueError("Sequence contains no hands")
    active = np.flatnonzero(present.any(axis=1))
    raw = raw[active[0]:active[-1] + 1]
    present = present[active[0]:active[-1] + 1]
    if len(raw) < 4:
        raise ValueError("Too few hand-visible frames")
    wrist = raw[:, :, :1, :]
    scale = np.linalg.norm(raw[:, :, 9, :2] - raw[:, :, 0, :2], axis=-1)
    local = (raw - wrist) / np.maximum(scale[..., None, None], 0.025)
    local *= present[..., None, None]
    # Local handshape + absolute image wrist position + missing-hand mask.
    features = np.concatenate((local.reshape(len(raw), 2, 63),
                               raw[:, :, 0, :2] * 2,
                               present[..., None].astype(np.float32)), axis=2)
    features = features.reshape(len(raw), -1)
    source = np.linspace(0, 1, len(raw))
    target = np.linspace(0, 1, STEPS)
    return np.stack([np.interp(target, source, col) for col in features.T], axis=1)


class SequenceModel:
    def __init__(self, labels, templates, thresholds, margin=0.02):
        self.labels = np.asarray(labels, dtype=str)
        self.templates = np.asarray(templates, dtype=np.float32)
        self.thresholds = dict(thresholds)
        self.margin = float(margin)
        if (not len(self.labels) or self.templates.shape != (len(self.labels), STEPS, 132)
                or not np.isfinite(self.templates).all()
                or set(self.labels) != set(self.thresholds)
                or any(not np.isfinite(v) or v <= 0 for v in self.thresholds.values())
                or not np.isfinite(self.margin) or self.margin < 0):
            raise ValueError("Invalid sequence model")

    @classmethod
    def fit(cls, samples):
        labels = [s["label"] for s in samples]
        if len(set(labels)) < 2:
            raise ValueError("Record at least two word classes")
        templates = np.stack([encode(s["frames"]) for s in samples])
        thresholds = {}
        for label in sorted(set(labels)):
            group = templates[np.asarray(labels) == label]
            if len(group) < 3:
                raise ValueError(f"Need >=3 training clips for {label}")
            distances = np.sqrt(np.mean((group[:, None] - group[None, :]) ** 2, axis=(2, 3)))
            np.fill_diagonal(distances, np.inf)
            thresholds[label] = max(0.10, float(np.quantile(distances.min(axis=1), .95)) * 1.5)
        return cls(labels, templates, thresholds)

    def predict(self, frames):
        query = encode(frames)
        distances = np.sqrt(np.mean((self.templates - query) ** 2, axis=(1, 2)))
        ranked = sorted((float(distances[self.labels == label].min()), label)
                        for label in set(self.labels))
        distance, label = ranked[0]
        separation = ranked[1][0] - distance if len(ranked) > 1 else float("inf")
        accepted = distance <= self.thresholds[label] and separation >= self.margin
        return {"label": label if accepted else None, "candidate": label,
                "distance": distance, "margin": separation}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, version=VERSION, labels=self.labels,
                            templates=self.templates,
                            thresholds=json.dumps(self.thresholds), margin=self.margin)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != VERSION:
                raise ValueError("Unsupported model version")
            return cls(data["labels"], data["templates"],
                       json.loads(str(data["thresholds"])), float(data["margin"]))


class ClipSegmenter:
    """Explicit word boundary: hands out of view for gap_s seconds."""
    def __init__(self, gap_s=0.35, min_s=0.25, max_s=5.0):
        self.gap_s, self.min_s, self.max_s = gap_s, min_s, max_s
        self.reset()

    def reset(self):
        self.frames = []
        self.start = self.last_seen = None
        self.blocked = False

    def update(self, frame, now):
        if np.any(frame):
            if self.blocked:
                self.last_seen = now
                return None
            if self.start is None:
                self.start = now
            self.last_seen = now
            self.frames.append(frame)
            if now - self.start > self.max_s:
                self.frames = []
                self.blocked = True
            return None
        if self.last_seen is None or now - self.last_seen < self.gap_s:
            return None
        frames = self.frames
        duration = self.last_seen - self.start if self.start is not None else 0
        self.reset()
        return frames if len(frames) >= 4 and duration >= self.min_s else None


class SentenceComposer:
    def __init__(self, rules):
        self.rules = {tuple(r["tokens"]): r["text"] for r in rules}
        self.clear()

    def clear(self):
        self.tokens = []
        self.text = ""

    def add(self, label):
        if len(self.tokens) >= 12:
            return False
        self.tokens.append(label)
        self.text = ""
        return True

    def finish(self):
        self.text = self.rules.get(tuple(self.tokens), "지원하지 않는 단어 조합입니다.")
        return self.text
