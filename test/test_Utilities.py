import cv2
import numpy as np

from test.setup import EvaluationSetup
import unittest
import src.project.Utilities as util


class Utilities(unittest.TestCase):
    def setUp(self):
        self.setup = EvaluationSetup("resources/inputs/", "resources/outputs/")

    def tearDown(self):
        self.setup = None

    def test_loadImage_png(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = util.loadImage("resources/inputs/test_loadImage.png")
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))

    def test_loadImage_jpg(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        self.actual = util.loadImage("resources/inputs/test_loadImage.jpg")
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))


    def test_getDFT(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        input = self.setup.getInputImage('chaplin.png')
        self.actual = util.getDFT(input)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_getImage(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        input = self.setup.getInputMatrix('chaplin_dft.npy')

        self.actual = util.getImage(input)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))


    def test_applyMask(self):
        expected = self.setup.getExpectedOutputMatrix(self._testMethodName)
        dft_image = self.setup.getInputMatrix('chaplin_dft.npy')
        mask = self.setup.getInputMatrix('mask.npy')

        self.actual = util.applyMask(dft_image, mask)
        self.assertTrue(self.setup.matrixEqual(expected, self.actual))


    def test_getImage_from_noisy_dft(self):
        expected = self.setup.getExpectedOutput(self._testMethodName)
        input = self.setup.getInputMatrix('noisy_dft.npy')
        self.actual = util.getImage(input)
        self.assertTrue(self.setup.imagesEqual(expected, self.actual))


if __name__ == '__main__':
    unittest.main()


