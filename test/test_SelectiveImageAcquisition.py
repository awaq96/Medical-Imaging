from test.setup import EvaluationSetup
import unittest
import src.project.SelectiveImageAcquisition as aqc

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import cv2

class SelectiveImageAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.setup = EvaluationSetup("resources/inputs/", "resources/outputs/")
        self.mask_size = (1000, 1000)
        self.actual = None

    def tearDown(self):
        self.setup = None

    def test_cartesian_pattern_basic(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.cartesianPattern(self.mask_size, 0.01)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_cartesian_pattern_5_percent(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.cartesianPattern(self.mask_size, 0.05)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_cartesian_pattern_10_percent(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.cartesianPattern(self.mask_size, 0.1)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_cartesian_pattern_30_percent(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.cartesianPattern(self.mask_size, 0.3)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_cartesian_pattern_60_percent(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.cartesianPattern(self.mask_size, 0.6)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_circle_acquisition_pattern_when_radius_is_small(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.circlePattern(self.mask_size, 100)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_circle_acquisition_pattern_when_radius_is_half(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.circlePattern(self.mask_size, 500)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_circle_acquisition_pattern_when_radius_is_same_as_image(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.circlePattern(self.mask_size, 1000)

        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_circle_acquisition_pattern_when_radius_is_greater_than_image(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.circlePattern(self.mask_size, 1200)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_ellipse_acquisition_pattern_centered_vertical_ellipse(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.ellipsePattern(self.mask_size, 100, 200, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_ellipse_acquisition_pattern_vertical_ellipse_bigger_than_image(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.ellipsePattern(self.mask_size, 100, 1500, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_ellipse_acquisition_pattern_centered_horizontal_ellipse(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.ellipsePattern(self.mask_size, 200, 100, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_ellipse_acquisition_pattern_horizontal_ellipse_bigger_than_image(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.ellipsePattern(self.mask_size, 1500, 100, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_band_acquisition_pattern_width_is_small(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 30, 500, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_width_is_large(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 500, 1000, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_invalid_negative_width(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, -5, 1000, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_width_larger_than_image(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 2000, 1000, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_zero(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 0)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_10(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 10)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_30(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 30)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_45(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 45)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_90(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 90)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_180(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 180)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_270(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 270)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_360(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 360)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_370(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 370)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_band_acquisition_pattern_angle_400(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.bandPattern(self.mask_size, 10, 500, 400)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_radial_acquisition_pattern_basic(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.radialPattern(self.mask_size, 10)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_radial_acquisition_pattern_one_ray(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.radialPattern(self.mask_size, 1)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_radial_acquisition_pattern_5(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.radialPattern(self.mask_size, 5)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_radial_acquisition_pattern_180(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.radialPattern(self.mask_size, 180)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_radial_acquisition_pattern_360(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.radialPattern(self.mask_size, 360)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_spiral_acquisition_pattern_sparsity_1(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.spiralPattern(self.mask_size, 1)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_spiral_acquisition_pattern_sparsity_3(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.spiralPattern(self.mask_size, 3)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_spiral_acquisition_pattern_sparsity_5(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.spiralPattern(self.mask_size, 5)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))

    def test_spiral_acquisition_pattern_sparsity_10(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        self.actual = aqc.spiralPattern(self.mask_size, 10)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


if __name__ == '__main__':
    unittest.main()
