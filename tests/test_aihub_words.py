import tempfile
import unittest
from pathlib import Path
import numpy as np
from core.aihub_words import AIHubWordModel, features


def sample(kind):
    frame = np.ones((8, 67, 3), dtype=np.float32)
    frame[:, :, :2] = 0
    frame[:, 44, :2] = [1, 0]
    frame[:, 47, :2] = [-1, 0]
    for hand in (0, 21):
        for joint in range(21):
            frame[:, hand+joint, :2] = [joint * .1, (joint % 4) * (.1 + .2 * kind)]
    return frame


class AIHubTests(unittest.TestCase):
    def test_normalization(self):
        source = sample(0)
        shifted = source.copy()
        shifted[:, :, :2] = source[:, :, :2] * 200 + [500, 300]
        np.testing.assert_allclose(features(source), features(shifted), atol=1e-5)

    def test_roundtrip_and_predictions(self):
        model = AIHubWordModel.fit([{'label': str(i), 'frames': sample(i)} for i in range(2)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.npz'
            model.save(path)
            model = AIHubWordModel.load(path)
            self.assertEqual(model.predict(sample(0))['label'], '0')
            self.assertEqual(model.predict(sample(1))['label'], '1')

    def test_reject_missing_body(self):
        source = sample(0)
        source[:, 44, 2] = 0
        with self.assertRaises(ValueError): features(source)


if __name__ == '__main__': unittest.main()
