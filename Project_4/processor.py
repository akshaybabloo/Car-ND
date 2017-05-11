import helper


class DetectLanes:
    """
    Detect lanes and process them.
    """

    def __init__(self, perspective_source, perspective_destination, camera_calibration, number_frame=1):

        # Creates and object of perspective.
        self.perspective_transform = helper.PerspectiveTransformer(perspective_source, perspective_destination)

        self.camera_calibration = camera_calibration
        self.number_frame = number_frame

        # Boolean values for detecting left and right line
        self.left_line = None
        self.right_line = None

    def generate_frame(self):
        pass

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
        new_left = helper.Line(y=left_line[0], x=right_line[1])
        new_right = helper.Line(y=left_line[0], x=right_line[1])
        is_parallel = new_left.is_current_fit_parallel(new_right, threshold=parallel_thresh)
        dist = new_left.get_current_fit_distance(new_right)
        is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]

        return is_parallel & is_plausible_dist
