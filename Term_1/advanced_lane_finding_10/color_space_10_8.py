"""
HLS and Color Threshold
-----------------------

You've now seen that various color thresholds can be applied to find the lane lines in images. Here we'll explore 
this a bit further and look at a couple examples to see why a color space like HLS can be more robust.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run():
    """
    Run different HLS and its thresholds.
    """
    image = mpimg.imread('test6.jpg')

    # Converting original to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold for original image
    thresh = (180, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    thresh_2 = (200, 255)
    binary_2 = np.zeros_like(red)
    binary_2[(red > thresh_2[0]) & (red <= thresh_2[1])] = 1

    # Converting image to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Splitting HSL
    hue = hls[:, :, 0]
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]

    # Threshold for saturation
    thresh_3 = (90, 255)
    binary_3 = np.zeros_like(saturation)
    binary_3[(saturation > thresh_3[0]) & (saturation <= thresh_3[1])] = 1

    # Threshold for Hue
    thresh_4 = (15, 100)
    binary_4 = np.zeros_like(hue)
    binary_4[(hue > thresh_4[0]) & (hue <= thresh_4[1])] = 1

    # -------------------- Figure -----------------------

    f = plt.figure()

    size_x, size_y = (4, 4)

    f.add_subplot(size_x, size_y, 1)
    plt.imshow(image)
    plt.title("Original")

    f.add_subplot(size_x, size_y, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray")

    f.add_subplot(size_x, size_y, 3)
    plt.imshow(binary, cmap='gray')
    plt.title("Threshold of ({}, {})".format(thresh[0], thresh[1]))

    f.add_subplot(size_x, size_y, 4)
    plt.imshow(red, cmap='gray')
    plt.title("Red")

    f.add_subplot(size_x, size_y, 5)
    plt.imshow(green, cmap='gray')
    plt.title("Green")

    f.add_subplot(size_x, size_y, 6)
    plt.imshow(blue, cmap='gray')
    plt.title("Blue")

    f.add_subplot(size_x, size_y, 7)
    plt.imshow(binary_2, cmap='gray')
    plt.title("Threshold of Red color")

    f.add_subplot(size_x, size_y, 8)
    plt.imshow(hue, cmap='gray')
    plt.title("Hue")

    f.add_subplot(size_x, size_y, 9)
    plt.imshow(lightness, cmap='gray')
    plt.title("Lightness")

    f.add_subplot(size_x, size_y, 10)
    plt.imshow(saturation, cmap='gray')
    plt.title("Saturation")

    f.add_subplot(size_x, size_y, 11)
    plt.imshow(binary_3, cmap='gray')
    plt.title("Threshold of saturation")

    f.add_subplot(size_x, size_y, 12)
    plt.imshow(binary_4, cmap='gray')
    plt.title("Threshold of hue")

    plt.show()


if __name__ == '__main__':
    run()
