# Use this file as you wish to generate the images needed to answer the report
import cv2
import numpy as np
import sys

sys.path.append("..")
import project.Utilities as util
import project.SelectiveImageAcquisition as acq
from argparse import ArgumentParser

if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument("-i", "--image", dest="image", help="Specify the image to filter")
    parser.add_argument("-m", "--mask_type", dest="mask_type", help="Specify the name of the mask to use")
    parser.add_argument("-p", "--percentage", dest="percent", help="Specify the percentage of trajectory")
    parser.add_argument("-r", "--radius", dest="radius", help="Specify the radius")
    parser.add_argument("-maj", "--maj_axis", dest="maj_axis", help="Specify the major axis")
    parser.add_argument("-min", "--min_axis", dest="min_axis", help="Specify the minor axis")
    parser.add_argument("-a", "--angle", dest="angle", help="Specify the angle")
    parser.add_argument("-w", "--width", dest="width", help="Specify the width")
    parser.add_argument("-l", "--length", dest="length", help="Specify the length")
    parser.add_argument("-rc", "--raycount", dest="ray_count", help="Specify the ray count")
    parser.add_argument("-s", "--sparsity", dest="sparsity", help="Specify the sparsity")
    args = parser.parse_args()

    img = util.loadImage(args.image)
    mask_shape = [len(img), len(img[0])]

    if args.mask_type == "cartesianPattern":
        mask = acq.cartesianPattern(mask_shape, float(args.percent))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Cartesian_" + args.percent + "_percent.png"
        save_mask_name = "Cartesian_Mask_" + args.percent + "_percent.png"
        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        # python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m cartesianPattern -p .5

    if args.mask_type == "circlePattern":
        mask = acq.circlePattern(mask_shape, int(args.radius))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Circle" + args.radius + "_radius.png"
        save_mask_name = "Circle_Mask_" + args.radius + "_radius.png"
        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        # python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m circlePattern -r 100

    if args.mask_type == "ellipsePattern":
        mask = acq.ellipsePattern(mask_shape, int(args.maj_axis), int(args.min_axis), int(args.angle))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Ellipse" + args.angle + "_angle.png"
        save_mask_name = "Ellipse_Mask_" + args.angle + "_angle.png"
        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        # python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m ellipsePattern -maj 200 -min 100 -a 45

    if args.mask_type == "bandPattern":
        mask = acq.bandPattern(mask_shape, int(args.width), int(args.length), int(args.angle))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Band" + args.angle + "_angle_" + "_width_" + args.width +".png"
        save_mask_name = "Band_Mask_" + args.angle + "_angle.png"
        save_fft_name = "Band_FFT_" + args.angle + "_angle.png"
        magnitude_spectrum = 20 * np.log(np.abs(fft_image))

        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        util.saveImage(save_fft_name, magnitude_spectrum)

        # python3 ImageAcquisitionDriver.py -i images/brain.png -m bandPattern -w 10 -l 100 -a 90
        # python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m bandPattern -w 100 -l 100 -a 35


    if args.mask_type == "radialPattern":
        mask = acq.radialPattern(mask_shape, int(args.ray_count))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Radial" + args.ray_count + "_count.png"
        save_mask_name = "Radial_Mask_" + args.ray_count + "_count.png"
        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        # python3 ImageAcquisitionDriver.py -i images/brain.png -m radialPattern -rc 360

    if args.mask_type == "spiralPattern":
        mask = acq.spiralPattern(mask_shape, int(args.sparsity))
        norm_mask = util.normalizeImage(mask)
        fft_image = util.getDFT(img)
        convolution = util.applyMask(fft_image, norm_mask)
        filtered = util.getImage(convolution)
        saved_image_name = "Spiral" + args.sparsity + "_sparsity.png"
        save_mask_name = "Spiral_Mask_" + args.sparsity + "_sparsity.png"
        util.saveImage(save_mask_name, norm_mask)
        util.saveImage(saved_image_name, filtered)
        # python3 ImageAcquisitionDriver.py -i images/brain.png -m spiralPattern -s 10
