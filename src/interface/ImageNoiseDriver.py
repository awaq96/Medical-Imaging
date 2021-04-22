# Use this file as you wish to generate the images needed to answer the report
import sys
sys.path.append("..")
import project.Utilities as util
import project.ImageSynthesisNoise as im
import numpy as np
from argparse import ArgumentParser


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", dest="image", help="Specify the file path of the image to filter")
    parser.add_argument("-m", "--mask_type", dest="mask_type", help="Specify the name of the mask to use")
    parser.add_argument("-c", "--cutoff", dest="cutoff", help="Specify the cutoff frequency")
    parser.add_argument("-o", "--order", dest="order", help="Specify the order")
    parser.add_argument("-t", "--thickness", dest="thickness", help="Specify the thickess")
    parser.add_argument("-matrix", "--matrix", dest="matrix_name", help= "Specify the matrix file path")


    args = parser.parse_args()

    if args.matrix_name is None:
        img = util.loadImage(args.image)
        mask_shape = [len(img), len(img[0])]



    if args.mask_type == "idealLowpassFilter":
        mask = im.idealLowpassFilter(mask_shape, int(args.cutoff))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Ideal_Lowpass_" + args.cutoff + "cutoff.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m idealLowpassFilter -c 100

    if args.mask_type == "idealHighpassFilter":
        mask = im.idealHighpass(mask_shape, int(args.cutoff))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Ideal_Highpass_" + args.cutoff + "cutoff.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m idealHighpassFilter -c 100

    if args.mask_type == "gaussianLowpassFilter":
        mask = im.gaussianLowpassFilter(mask_shape, int(args.cutoff))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Gaussian_Lowpass" + args.cutoff + "cutoff.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m gaussianLowpassFilter -c 100

    if args.mask_type == "gaussianHighpassFilter":
        mask = im.gaussianHighpassFilter(mask_shape, int(args.cutoff))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Gaussian_Highpass" + args.cutoff + "cutoff.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m gaussianHighpassFilter -c 100

    if args.mask_type == "butterworthLowpassFilter":
        mask = im.butterworthLowpassFilter(mask_shape, int(args.cutoff), int(args.order))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Butterworth_Lowpass_" + args.cutoff + "_cutoff_" + args.order + "_order.png"
        util.saveImage(saved_image_name, filtered)
        snr = signaltonoise(filtered, axis=None)
        print(snr)

        # python3 ImageNoiseDriver.py -i images/brain.png -m butterworthLowpassFilter -c 100 -o 50

    if args.mask_type == "butterworthHighpassFilter":
        mask = im.butterworthHighpassFilter(mask_shape, int(args.cutoff), int(args.order))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Butterworth_Highpass" + args.cutoff + "_cutoff_" + args.order + "_order.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m butterworthHighpassFilter -c 100 -o 50


    if args.mask_type == "ringLowpassFilter":
        mask = im.ringLowpassFilter(mask_shape, int(args.cutoff), int(args.thickness))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Ring_Lowpass" + args.cutoff + "_cutoff_" + args.thickness + "_thickness.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m ringLowpassFilter -c 50 -t 10


    if args.mask_type == "ringHighpassFilter":
        mask = im.ringHighpassFilter(mask_shape, int(args.cutoff), int(args.thickness))
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Ring_Highpass" + args.cutoff + "_cutoff_" + args.thickness + "_thickness.png"
        util.saveImage(saved_image_name, filtered)
        # python3 ImageNoiseDriver.py -i images/brain.png -m ringHighpassFilter -c 50 -t 10

    if args.matrix_name is not None:
        matrix = util.loadMatrix(args.matrix_name)
        orignal_image = matrix.astype(np.uint8)

        mask = im.ringLowpassFilter((len(matrix), len(matrix[0])), 1, 100)
        fft_image = util.getDFT(orignal_image)
        convolution = util.applyMask(fft_image, mask)
        filtered = util.getImage(convolution)

        saved_image_name = "noisyImageFiltered.png"
        util.saveImage(saved_image_name, filtered)
        util.saveImage("originalNoisyImage.png", orignal_image)
        # python3 ImageNoiseDriver.py -matrix images/noisyimage.npy
