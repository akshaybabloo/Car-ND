"""
Cluster of helper functions.
"""

import pickle
from glob import glob

import cv2
import numpy as np
from scipy.misc import imresize, imread
import os

# Constants
OUTPUT_DIR = os.path.abspath('output')
CALIBRATION_PATH_PICKLE = os.path.abspath('output' + os.sep + 'calibration.p')
CAL_IMAGE_PATH = os.path.abspath('camera_cal' + os.sep + 'calibration*.jpg')
ROWS, COLS = (6, 9)
CAL_IMAGE_SIZE = (720, 1280, 3)

"""
1. Calibrate the camera.
"""


class CalibrateCamera:
    def __init__(self, image_shape, calibration):
        """
        Removed lens distortion.
        
        Parameters
        ----------
        image_shape: tuple
            Width and height of the image.
        calibration: dict
            Calibrated image.
        """
        """
        Helper class to remove lens distortion from images
        
        :param image_shape: with and height of the image
        :param calibration: calibration object which can be retrieved from "get_camera_calibration()"
        """
        self.objpoints = calibration['objpoints']
        self.imgpoints = calibration['imgpoints']
        self.image_shape = image_shape

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_shape, None, None)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


def get_camera_calibration():
    """
    If the pickled image is present in the location given in `CALIBRATION_PATH_PICKLE`, then open the pickle and 
    return the data; if not call `calculate_camera_calibration`.
    """
    if not os.path.isfile(CALIBRATION_PATH_PICKLE):

        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        calibration = _calculate_camera_calibration(CAL_IMAGE_PATH, ROWS, COLS)
        with open(CALIBRATION_PATH_PICKLE, 'wb') as file:
            pickle.dump(calibration, file=file)
    else:
        with open(CALIBRATION_PATH_PICKLE, "rb") as file:
            calibration = pickle.load(file)

    return calibration


def _calculate_camera_calibration(path_pattern, rows, cols):
    """
    Based on the chessboard images located in `camera_cal`, calculate the camera calibration.
    
    Parameters
    ----------
    path_pattern: str
        Path pattern for Glob.
    rows: int
        Number of rows on chessboard.
    cols: int
        Number of columns on chess board.

    Returns
    -------
    calibration: dict
        A dictionary of `cv2.calibrateCamera`
    """

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob(path_pattern)
    cal_images = np.zeros((len(images), *CAL_IMAGE_SIZE), dtype=np.uint8)

    successful_count = 0
    for idx, fname in enumerate(images):
        img = imread(fname)
        if img.shape[0] != CAL_IMAGE_SIZE[0] or img.shape[1] != CAL_IMAGE_SIZE[1]:
            img = imresize(img, CAL_IMAGE_SIZE)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if ret:
            successful_count += 1

            objpoints.append(objp)
            imgpoints.append(corners)

            img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
            cal_images[idx] = img

    print("%s/%s camera calibration images processed." % (successful_count, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, CAL_IMAGE_SIZE[:-1], None, None)

    calibration = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'cal_images': cal_images,
                   'mtx': mtx,
                   'dist': dist,
                   'rvecs': rvecs,
                   'tvecs': tvecs}

    return calibration


"""
Other image utilities.
"""


def abs_sobel(img_ch, orient='x', kernel_size=3):
    """
    Takes the absolute values of Sobel derivative.
    
    Parameters
    ----------
    img_ch
    orient: str
        Orientation of the derivative. `x` or`y`
    kernel_size: int
        Kernel size.
        
    Returns
    -------
    abs_sobel: ndarray
        Absolute array of Sobel derivative.
    """

    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=kernel_size)
    abs_sobel = np.absolute(sobel)

    return abs_sobel


def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculates the magnitude of the gradient.
    
    Parameters
    ----------
    sobel_x
    sobel_y

    Returns
    -------

    """

    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)

if __name__ == '__main__':
    get_camera_calibration()
