"""AI Hub 2D hand/pose sequence baseline, separate from MediaPipe XYZ models."""
import json
from pathlib import Path
import numpy as np

STEPS = 32


def features(frames):
    """frames: T x 67 x 3 = left21, right21, pose25; third value is confidence."""
    data = np.asarray(frames, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (67, 3) or len(data) < 4:
        raise ValueError('Expected T>=4 frames of 67 (x,y,confidence) points')
    if not np.isfinite(data).all():
        raise ValueError('Non-finite coordinates')
    result = []
    for frame in data:
        pose = frame[42:]
        # OpenPose BODY_25: 2=right shoulder, 5=left shoulder.
        if min(pose[2, 2], pose[5, 2]) < .2:
            raise ValueError('Shoulder anchors missing')
        center = (pose[2, :2] + pose[5, :2]) / 2
        body_scale = np.linalg.norm(pose[2, :2] - pose[5, :2])
        if body_scale < 1e-6:
            raise ValueError('Invalid shoulder width')
        row = []
        for hand in (frame[:21], frame[21:42]):
            valid = hand[:, 2] >= .2
            if not (valid[0] and valid[9]) or valid.sum() < 15:
                row.extend([0.] * 45)
                continue
            scale = np.linalg.norm(hand[9, :2] - hand[0, :2])
            if scale < 1e-6:
                row.extend([0.] * 45)
                continue
            local = (hand[:, :2] - hand[0, :2]) / scale
            local[~valid] = 0
            row.extend(local.ravel())
            row.extend((hand[0, :2] - center) / body_scale)
            row.append(1.)
        result.append(row)
    result = np.asarray(result)
    if not np.any(result[:, [44, 89]]):
        raise ValueError('No valid hands')
    return np.stack([np.interp(np.linspace(0, 1, STEPS),
                              np.linspace(0, 1, len(result)), col)
                     for col in result.T], axis=1).astype(np.float32)


class AIHubWordModel:
    def __init__(self, labels, templates, max_distance=.35, min_margin=.03):
        self.labels = np.asarray(labels, dtype=str)
        self.templates = np.asarray(templates, dtype=np.float32)
        self.max_distance, self.min_margin = float(max_distance), float(min_margin)
        if (len(set(self.labels)) < 2 or self.templates.shape != (len(self.labels), STEPS, 90)
                or not np.isfinite(self.templates).all()):
            raise ValueError('Invalid AI Hub model')

    @classmethod
    def fit(cls, clips):
        return cls([c['label'] for c in clips], [features(c['frames']) for c in clips])

    def predict(self, frames):
        distances = np.sqrt(np.mean((self.templates - features(frames)) ** 2, axis=(1, 2)))
        ranking = sorted((float(distances[self.labels == label].min()), str(label))
                         for label in set(self.labels))
        distance, candidate = ranking[0]
        margin = ranking[1][0] - distance
        accepted = distance <= self.max_distance and margin >= self.min_margin
        return {'label': candidate if accepted else None, 'candidate': candidate,
                'distance': distance, 'margin': margin,
                'ranking': [{'label': label, 'distance': d} for d, label in ranking]}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, format='aihub-openpose-2d-v1', labels=self.labels,
                            templates=self.templates, max_distance=self.max_distance,
                            min_margin=self.min_margin)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != 'aihub-openpose-2d-v1':
                raise ValueError('Not an AI Hub 2D model')
            return cls(data['labels'], data['templates'], data['max_distance'], data['min_margin'])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict a saved AI Hub word clip')
    parser.add_argument('clip', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/ksl_aihub_first.npz'))
    args = parser.parse_args()
    clip = json.loads(args.clip.read_text(encoding='utf-8'))
    result = AIHubWordModel.load(args.model).predict(clip['frames'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
