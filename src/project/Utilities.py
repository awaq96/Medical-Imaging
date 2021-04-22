import cv2
import numpy as np


def loadImage(image_path):
    image = cv2.imread(image_path, 0)
    return image


def loadMatrix(filename):
    matrix = np.load(filename)
    return matrix


def saveImage(filename, image):
    return cv2.imwrite(filename, image)


def saveMatrix(filename, matrix):
    # return np.save(f)
    return cv2.imwrite(filename, matrix)


# map input image to values from 0 to 255"
def normalizeImage(image):
    normalized = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return normalized


# Remember: the DFT its a decomposition of signals
#  To be able to save it as an image you must convert it.
def writableDFT(dft_image):
    converted = np.fft.ifftshift(dft_image)
    converted = np.fft.ifft2(converted)
    return converted


# Use openCV to display your image"
# Remember: normalize binary masks and convert FFT matrices to be able to see and save them"
def displayImage(image):
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getDFT(image):
    image = np.fft.fft2(image)
    image = np.fft.fftshift(image)
    return image


# Confert from fft matrix to an image"
def getImage(dft_img):
    dft_img = np.fft.ifftshift(dft_img)
    dft_img = np.fft.ifft2(dft_img)
    image = post_process_image(dft_img)

    return image

# Both input values must be raw values"
def applyMask(image_dft, mask):
    return image_dft * mask


def signalToNoise():
    return False


#[Provide] Use this function to acomplish a good final image
def post_process_image(image):
    a = np.min(image)
    b = np.max(image)
    k = 255
    image = (image - a) * (k / (b - a))
    return image.astype('uint8')
