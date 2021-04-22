import cv2
import numpy as np
import src.project.Utilities as util


class EvaluationSetup:

    def __init__(self, inputpath, outputpath):
        self.pathInput = inputpath
        self.pathExpectOutput = outputpath
        self.testname = ""

    def normalizeImage(self, image):
        return image / np.max(image) * 255

    def getExpectedOutput(self, testname):
        self.testname = testname
        return util.loadImage(self.pathExpectOutput + self.testname + '.png')

    def getExpectedOutputMatrix(self, testname):
        self.testname = testname
        filename = self.pathExpectOutput + self.testname + '.npy'
        with open(filename, 'rb') as f:
            matrix = np.load(f, allow_pickle=True)
        return matrix

    def getInputImage(self, imagename):
        return util.loadImage(self.pathInput + imagename)

    def getInputMatrix(self, filename):
        with open(self.pathInput + filename, 'rb') as f:
            matrix = np.load(f)
        return matrix

    def imagesEqual(self, expected, actual):
        #First check images have same number of channels
        if len(actual.shape) != len(expected.shape):
            return False

        # First check if image dimensions are the same
        if len(actual.shape) == 1 \
            or expected.shape[0] != actual.shape[0] \
                or expected.shape[1] != actual.shape[1]:
            return False

        # Take image difference
        difference = cv2.subtract(expected, actual.astype(np.uint8))

        # check if image is single or triple channel
        if len(expected.shape) == 2:
            return cv2.countNonZero(difference) == 0
        elif len(expected.shape) == 3:
            r, g, b = cv2.split(expected)
            return cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0

        return False

    def matrixEqual(self, expected, actual):
        difference = expected - actual
        non_zero = np.sum(difference[difference != 0])
        return non_zero == 0

    def saveImage(self, image):
        filename = self.testname + '.png'
        return cv2.imwrite(self.pathExpectOutput + filename, image.astype(np.uint8))

    def saveMatrix(self, matrix):
        filename = self.testname + '.npy'
        with open(self.pathExpectOutput + filename, 'wb') as f:
            np.save(f, matrix, allow_pickle=True)


