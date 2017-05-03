import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Absolute of Sobel derivation.
    
    Parameters
    ----------
    img: Numpy array
        Image
    orient: str
        Direction of derivation, ``x`` or ``y`` axis.
    sobel_kernel: int
        Kernel number.
    thresh: tuple
        Lower and upper threshold.

    Returns
    -------
    Numpy Array:
        Binary array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Magnitude of threshold - Sobel derivation.
    
    Parameters
    ----------
    img: Numpy array
        Image
    sobel_kernel: int
        Kernel number 
    mag_thresh: tuple
        Lower and upper threshold.

    Returns
    -------
    Numpy Array:
        Binary array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Direction of threshold - Sobel derivation.
    
    Parameters
    ----------
    img: Numpy array
        Image
    sobel_kernel: int
        Kernel number
    thresh: tuple
        Lower and upper threshold.

    Returns
    -------
    Numpy Array:
        Binary array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def run():
    """
    Run the methods.
    """
    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    image = mpimg.imread('signs_vehicles_xygrad.png')

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(10, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(10, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    f = plt.figure()

    f.add_subplot(3, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    f.add_subplot(3, 3, 2)
    plt.imshow(gradx, cmap='gray')
    plt.title("Absolute of Sobel at x axis")

    f.add_subplot(3, 3, 3)
    plt.imshow(grady, cmap='gray')
    plt.title("Absolute of Sobel at y axis")

    f.add_subplot(3, 3, 4)
    plt.imshow(mag_binary, cmap='gray')
    plt.title("Gradient Magnitude")

    f.add_subplot(3, 3, 5)
    plt.imshow(dir_binary, cmap='gray')
    plt.title("Gradient Direction")

    f.add_subplot(3, 3, 6)
    plt.imshow(combined, cmap='gray')
    plt.title("Combined")

    plt.show()

if __name__ == '__main__':
    run()
