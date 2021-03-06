"""
Color and Gradient
------------------

At this point, it's okay to detect edges around trees or cars because these lines can be mostly filtered out by 
applying a mask to the image and essentially cropping out the area outside of the lane lines. It's most important 
that you reliably detect different colors of lane lines under varying degrees of daylight and shadow. 

You can clearly see which parts of the lane lines were detected by the gradient threshold and which parts were 
detected by the color threshold by stacking the channels and seeing the individual components. You can create a 
binary combination of these two images to map out where either the color or gradient thresholds were met. 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Pipeline for getting color and gradient.
    
    Parameters
    ----------
    img: Numpy array
        Image array.
    s_thresh: tuple
        A tuple of lower and upper threshold for color channel.
    sx_thresh: tuple
        A tuple of lower and upper threshold for ``x`` Sobel.

    Returns
    -------
    color_binary: Numpy array
        Binary values color.
    """
    img = np.copy(img)

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary


def run():
    """
    Running the pipeline and plotting it.
    """
    image = mpimg.imread('test6.jpg')

    result = pipeline(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if __name__ == '__main__':
    run()
