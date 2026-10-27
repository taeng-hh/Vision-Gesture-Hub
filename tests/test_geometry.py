import unittest
from core.geometry import normalize_to_roi, pinch_ratio, is_open_palm
from core.types import Hand, Landmark


class TestGeometry(unittest.TestCase):
    def test_open_palm_geometry_without_label(self):
        points = [Landmark(0.5, 0.8, 0) for _ in range(21)]
        for x, base, joint, tip in ((0.35, 5, 6, 8), (0.45, 9, 10, 12),
                                     (0.55, 13, 14, 16), (0.65, 17, 18, 20)):
            points[base] = Landmark(x, 0.6, 0)
            points[joint] = Landmark(x, 0.45, 0)
            points[tip] = Landmark(x, 0.25, 0)
        points[4] = Landmark(0.15, 0.6, 0)
        self.assertTrue(is_open_palm(Hand(tuple(points))))
        for tip in (8, 12, 16, 20):
            points[tip] = Landmark(0.5, 0.75, 0)
        self.assertFalse(is_open_palm(Hand(tuple(points))))

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
