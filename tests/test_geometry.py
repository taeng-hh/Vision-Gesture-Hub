import unittest
from core.geometry import normalize_to_roi, pinch_ratio
from core.types import Hand, Landmark


class TestGeometry(unittest.TestCase):
    def test_normalized_cursor(self):
        self.assertEqual(normalize_to_roi(0.5, 0.5), (0.5, 0.5))
        self.assertEqual(normalize_to_roi(0, 1), (0, 1))

    def test_pinch_ratio(self):
        points = [Landmark(0, 0, 0) for _ in range(21)]
        points[5] = Landmark(0, 0, 0)
        points[17] = Landmark(1, 0, 0)
        points[4] = Landmark(0.2, 0, 0)
        points[8] = Landmark(0.3, 0, 0)
        self.assertAlmostEqual(pinch_ratio(Hand(tuple(points))), 0.1)
