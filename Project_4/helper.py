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
2. Change the perspective
"""


class PerspectiveTransformer:
    """
    Helps to change the perspective of the image.
    """
    def __init__(self, src, dst):
        """
        Parameters
        ----------
        src: str
            Source coordinates.
        dst:
            Destination coordinates.
        """

        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """
        Transform the image using CV2's `warpPerspective`.
        
        Parameters
        ----------
        img: ndarray
            Image.

        Returns
        -------
        image: ndarray
            Transformed image.

        """
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        """
        Inverse transform the image using CV2's `warpPerspective`.
        
        Parameters
        ----------
        img: ndarray
            Image.

        Returns
        -------
        image: ndarray
            Transformed image.
        """
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


"""
3. Generate images.
"""


def generate_lane_mask(img, v_cutoff=0):
    """
    Generate a masking binary image with lane selected.

    Parameters
    ----------
    img: ndarray
        Image.
    v_cutoff:
        Image cutoff.

    Returns
    -------
    mask: ndarray
        A binary Numpy array of the masked image.

    """

    window = img[v_cutoff:, :, :]
    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    s_x = abs_sobel(gray, orient='x', kernel_size=3)
    s_y = abs_sobel(gray, orient='y', kernel_size=3)

    grad_dir = gradient_direction(s_x, s_y)
    grad_mag = gradient_magnitude(s_x, s_y)

    ylw = extract_yellow(window)
    highlights = extract_highlights(window[:, :, 0])

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
                        (s_y >= 25) & (s_y <= 255)) |
                       ((grad_mag >= 30) & (grad_mag <= 512) &
                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                       (ylw == 255) |
                       (highlights == 255)] = 1

    mask = binary_noise_reduction(mask, 4)

    return mask


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


def gradient_direction(sobel_x, sobel_y):
    """
    Calculates the direction of the gradient. NaN values cause by zero division will be replaced
    by the maximum value (np.pi / 2).
    
    Parameters
    ----------
    sobel_x
    sobel_y

    Returns
    -------

    """

    abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
    abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def extract_yellow(img):
    """
    Mask all yellow pixels.
    
    Parameters
    ----------
    img

    Returns
    -------
    mask: ndarray
        Masked image

    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    return mask


def extract_highlights(img, p=99.9):
    """
    Get the selected highlights from the image.
    
    Parameters
    ----------
    img
    p: float
        Percentile to compute

    Returns
    -------
    mask: ndarray
        Masked image
    """

    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask


def binary_noise_reduction(img, thresh):
    """
    
    Parameters
    ----------
    img
    thresh

    Returns
    -------
    img: ndarray
        Filtered image.
    """

    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


"""
4. Detect the lines on the road based on image generated.
"""


class Line:
    """
    Detect th lines on th road.
    """
    def __init__(self, n_frames=1, x=None, y=None):
        """
        Parameters
        ----------
        n_frames: int
            Number of frames to smooth
        x: int
            `X` coordinates
        y: int
            `Y` coordinates
        """

        # Frame memory
        self.n_frames = n_frames
        # was the line detected in the last iteration?
        self.detected = False
        # number of pixels added per frame
        self.n_pixel_per_frame = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # Polynom for the current coefficients
        self.current_fit_poly = None
        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        """
        Update the lines.
        Parameters
        ----------
        x: list
            List of `X` values
        y: list
            List of `Y` values
        """

        assert len(x) == len(y), 'x and y have to be the same size'

        self.allx = x
        self.ally = y

        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]

        self.bestx = np.mean(self.recent_xfitted)

        self.current_fit = np.polyfit(self.allx, self.ally, 2)

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_current_fit_parallel(self, other_line, threshold=(0, 0)):
        """
        
        Parameters
        ----------
        other_line
        threshold: tuple
            Threshold for 

        Returns
        -------

        """

        first_coefi_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coefi_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])

        is_parallel = first_coefi_dif < threshold[0] and second_coefi_dif < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):
        """
        Gets the distance between the current fit polynomials of two lines
        
        Parameters
        ----------
        other_line

        Returns
        -------

        """

        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))

    def get_best_fit_distance(self, other_line):
        """
        Gets the distance between the best fit polynomials of two lines
        
        Parameters
        ----------
        other_line

        Returns
        -------
        absolute: ndarray
            Absolute value.
        """

        return np.abs(self.best_fit_poly(719) - other_line.best_fit_poly(719))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import misc
    import os

    image = misc.imread('test_images' + os.sep + 'test2.jpg')
    masked_image = generate_lane_mask(image)

    plt.imshow(masked_image, cmap='gray')
    plt.show()
