from test.setup import EvaluationSetup
import unittest
import src.project.ImageSynthesisNoise as noise

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class ImageSynthesisNoise(unittest.TestCase):
    def setUp(self):
        self.setup = EvaluationSetup("resources/inputs/", "resources/outputs/")
        self.emptymask = (200, 200)
        self.actual = None

    def tearDown(self):
        self.setup = None

    def test_idealLowpassFilter_low_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealLowpassFilter(self.emptymask, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_idealLowpassFilter_high_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealLowpassFilter(self.emptymask, 150)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_idealLowpassFilter_half_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealLowpassFilter(self.emptymask, 100)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_idealHighpassFilter_low_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealHighpassFilter(self.emptymask, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_idealHighpassFilter_high_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealHighpassFilter(self.emptymask, 150)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_idealHighpassFilter_half_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.idealHighpassFilter(self.emptymask, 100)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianLowpassFilter_low_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianLowpassFilter(self.emptymask, 20)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianLowpassFilter_high_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianLowpassFilter(self.emptymask, 180)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianLowpassFilter_half_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianLowpassFilter(self.emptymask, 100)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianHighpassFilter_low_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianHighpassFilter(self.emptymask, 20)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianHighpassFilter_high_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianHighpassFilter(self.emptymask, 180)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_gaussianHighpassFilter_half_width(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.gaussianHighpassFilter(self.emptymask, 100)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_small_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 10, 1)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_small_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 10, 5)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_half_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 50, 3)
        self.actual = self.setup.normalizeImage(self.actual)

        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_half_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 50, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_large_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 100, 3)
        self.actual = self.setup.normalizeImage(self.actual)

        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringLowpassFilter_large_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringLowpassFilter(self.emptymask, 100, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))


    def test_ringHighpassFilter_small_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 10, 1)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringHighpassFilter_small_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 10, 5)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringHighpassFilter_half_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 50, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringHighpassFilter_half_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 50, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringHighpassFilter_large_thin_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 100, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_ringHighpassFilter_large_thick_ring(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.ringHighpassFilter(self.emptymask, 100, 10)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthLowpassFilter_low_width_low_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthLowpassFilter(self.emptymask, 10, 1)
        self.actual = self.setup.normalizeImage(self.actual)

        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthLowpassFilter_low_width_high_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthLowpassFilter(self.emptymask, 10, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthLowpassFilter_high_width_low_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthLowpassFilter(self.emptymask, 50, 1)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthLowpassFilter_high_width_high_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthLowpassFilter(self.emptymask, 50, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthHighpassFilter_low_width_low_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthHighpassFilter(self.emptymask, 10, 1)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthHighpassFilter_low_width_high_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthHighpassFilter(self.emptymask, 10, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthHighpassFilter_high_width_low_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthHighpassFilter(self.emptymask, 50, 1)
        self.actual = self.setup.normalizeImage(self.actual)

        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_butterworthHighpassFilter_high_width_high_order(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = noise.butterworthHighpassFilter(self.emptymask, 50, 3)
        self.actual = self.setup.normalizeImage(self.actual)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

if __name__ == '__main__':
    unittest.main()
