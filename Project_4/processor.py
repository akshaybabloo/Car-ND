import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import helper


class DetectLanes:
    """
    Detect lanes and process them.
    """

    def __init__(self, perspective_source, perspective_destination, camera_calibration=None, number_frame=1,
                 line_segments=10, transform_offset=0, verbose=False):

        # Creates and object of perspective.
        self.perspective_transform = helper.PerspectiveTransformer(perspective_source, perspective_destination)

        self.camera_calibration = camera_calibration
        self.number_frame = number_frame
        self.line_segments = line_segments
        self.image_offset = transform_offset
        self.verbose = verbose

        # Boolean values for detecting left and right line
        self.left_line = None
        self.right_line = None
        self.center_poly = None

        # UI calibrations
        self.curvature = 0.0
        self.offset = 0.0
        self.offset = 0.0

        self.dists = []

    def generate_frame(self, frame):
        """
        Apply line detection to each frame.
        
        Parameters
        ----------
        frame: ndarray
            Image.

        Returns
        -------
        copy_frame: ndarray
            Processed image.
        """

        copy_frame = np.copy(frame)

        # Get undistorted image if camera_calibration is not None. helper.CalibrateCamera() should be given
        if self.camera_calibration is not None:
            frame = self.camera_calibration.undistort(frame)

        # Add all the filters to generate a masked image.
        frame = helper.generate_lane_mask(frame, 400)

        # Get the birds eye view.
        frame = self.perspective_transform.transform(frame)

        # Reset on every call of this method.
        left_detected = False
        right_detected = False
        left_x = []
        left_y = []
        right_x = []
        right_y = []

        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(frame, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(frame, self.right_line.best_fit_poly, self.line_segments)

            left_detected, right_detected = self._check_lines(left_x, left_y, right_x, right_y)

        # If no lanes are found a histogram search will be performed
        if not left_detected:
            left_x, left_y = histogram_lane_detection(frame, self.line_segments,
                                                      (self.image_offset, frame.shape[1] // 2), h_window=7)
            left_x, left_y = outlier_removal(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(frame, self.line_segments,
                                                        (frame.shape[1] // 2, frame.shape[1] - self.image_offset),
                                                        h_window=7)
            right_x, right_y = outlier_removal(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self._check_lines(left_x, left_y, right_x, right_y)

        # Updated left lane information.
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = helper.Line(self.number_frame, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = helper.Line(self.number_frame, right_y, right_x)

        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = helper.calc_curvature(self.center_poly)
            self.offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700

            self._draw_lane(copy_frame)
            self._draw_ui(copy_frame)

        return copy_frame

    def _check_lines(self, left_x, left_y, right_x, right_y):
        """
        Compares left and right lines to its previously predicted lines.
        
        Parameters
        ----------
        left_x
        left_y
        right_x
        right_y

        Returns
        -------
        tuple: tuple
            A tuple of boolean values.
        """

        left_detected = False
        right_detected = False

        if is_line((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if is_line((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if is_line((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected

    def _draw_ui(self, img3):
        """
        Draws UI with all the information of what is happening.
        
        Parameters
        ----------
        img3: ndarray
            Image.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(img3, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img3, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)

    def _draw_lane(self, img4):

        overlay = np.zeros([*img4.shape])  # Converts tuple to list
        mask = np.zeros([img4.shape[0], img4.shape[1]])  # An array of zero to mask

        # lane area
        lane_area = calculate_lane_area((self.left_line, self.right_line), img4.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = self.perspective_transform.inverse_transform(mask)

        overlay[mask == 1] = (255, 128, 0)
        selection = (overlay != 0)
        img4[selection] = img4[selection] * 0.3 + overlay[selection] * 0.7

        # left and right line
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255)
        mask = self.perspective_transform.inverse_transform(mask)
        img4[mask == 255] = (255, 200, 2)

        if self.verbose:
            mask = self.perspective_transform.transform(img4)
            plt.imshow(mask)
            plt.show()


def draw_poly(img5, poly, steps, color, thickness=10, dashed=False):
    """
    Draw poly on the image.
    
    Parameters
    ----------
    img5
    poly
    steps
    color
    thickness
    dashed

    Returns
    -------
    img5: ndarray
        Image.
    """

    img_height = img5.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img5 = cv2.line(img5, end_point, start_point, color, thickness)

    return img5


def is_line(left_line, right_line, parallel_thresh=(0.0003, 0.55), dist_thresh=(350, 460)):
    """
    Check if the current lanes are what it seems to be based on distance and curvature.
    
    Parameters
    ----------
    left_line
    right_line
    parallel_thresh
    dist_thresh

    Returns
    -------

    """

    if len(left_line[0]) < 3 or len(right_line[0]) < 3:
        return False
    else:
        new_left = helper.Line(y=left_line[0], x=left_line[1])
        new_right = helper.Line(y=right_line[0], x=right_line[1])
        is_parallel = new_left.is_current_fit_parallel(new_right, threshold=parallel_thresh)
        dist = new_left.get_current_fit_distance(new_right)
        is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]

        return is_parallel & is_plausible_dist


def detect_lane_along_poly(img1, poly, steps):
    """
    Using moving window to select pixels.
    
    Parameters
    ----------
    img1
    poly
    steps

    Returns
    -------
    all: tuple
        Detected `x` and `y` pixel.
    """

    pixels_per_step = img1.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img1.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img1, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def histogram_lane_detection(img6, steps, search_window, h_window):
    """
    Applying sliding histogram to find the lines.
    
    Parameters
    ----------
    img6
    steps
    search_window
    h_window

    Returns
    -------
    all: tuple
        Detected `x` and `y` pixels.
    """

    all_x = []
    all_y = []
    masked_img = img6[:, search_window[0]:search_window[1]]
    pixels_per_step = img6.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))

        highest_peak = highest_n_peaks(histogram_smooth, peaks, n=1, threshold=5)
        if len(highest_peak) == 1:
            highest_peak = highest_peak[0]
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def get_pixel_in_window(img2, x_center, y_center, size):
    """
    Gets all the pixels in the window.
    
    Parameters
    ----------
    img2: ndarray
        Binary image
    x_center
    y_center
    size

    Returns
    -------
    all: tuple
        Selected `x` and `y` pixels.

    """

    half_size = size // 2
    window = img2[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size
    del img2
    return x, y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    """
    Gets the highest peaks of histogram.
    
    Parameters
    ----------
    histogram
    peaks
    n
    threshold

    Returns
    -------
    x: list
        Peaks of histogram.
    """

    if len(peaks) == 0:
        return []

    peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
    peak_list = sorted(peak_list, key=lambda _x: _x[1], reverse=True)

    if len(peak_list) == 0:
        return []

    x, y = zip(*peak_list)
    x = list(x)

    if len(peak_list) < n:
        return x

    return x[:n]


def outlier_removal(x, y, q=5):
    """
    Removes horizontal outliers based on a given percentile.
    
    Parameters
    ----------
    x
    y
    q

    Returns
    -------
    ndarray: tuple
        cleaned coordinates (x, y)
    """

    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]


def calculate_lane_area(lanes, area_height, steps):
    """
    Pixel coordinates between two lines.
    
    Parameters
    ----------
    lanes
    area_height
    steps

    Returns
    -------
    concat: ndarray
        All the pixels between the two given lines.
    """

    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = area_height // steps
        start = area_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)

# if __name__ == '__main__':
#     HIST_STEPS = 10
#     OFFSET = 250
#     FRAME_MEMORY = 5
#     SRC = np.float32([
#         (132, 703),
#         (540, 466),
#         (740, 466),
#         (1147, 703)])
#
#     DST = np.float32([
#         (SRC[0][0] + OFFSET, 720),
#         (SRC[0][0] + OFFSET, 0),
#         (SRC[-1][0] - OFFSET, 0),
#         (SRC[-1][0] - OFFSET, 720)])
#
#     cam_calibration = helper.get_camera_calibration()
#     img = plt.imread('test_images/test1.jpg')
#     cam_calibrator = helper.CalibrateCamera(img[:, :, 0].shape[::-1], cam_calibration)
#     ld = DetectLanes(SRC, DST, number_frame=FRAME_MEMORY, camera_calibration=cam_calibrator, transform_offset=OFFSET)
#
#     plt.imshow(ld.generate_frame(img))
#     plt.show()
