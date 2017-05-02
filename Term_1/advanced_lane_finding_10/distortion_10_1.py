import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run():
    """
    Count number of points where two points of white and black meets in x and y direction.
    
    Returns
    -------

    """
    # prepare object points
    nx = 0
    ny = 0

    # Make a list of calibration images
    fname = 'calibration_test.png'
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    run()
