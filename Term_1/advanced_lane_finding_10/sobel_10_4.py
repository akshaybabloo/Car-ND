import cv2
import numpy as np
import matplotlib.pyplot as plt


def run():
    img = cv2.imread('curved-lane.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # (1,0) denotes the x direction
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # (0,1) denotes the y direction
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))  # convert absolute value to 8-bit

    thresh_min = 20
    thresh_max = 100

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    plt.imshow(sxbinary, cmap='gray')
    plt.show()

if __name__ == '__main__':
    run()
