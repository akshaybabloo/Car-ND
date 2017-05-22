import cv2
import helper
import numpy as np
import old_process.helper as hp

prev_left_coeffs = None
prev_right_coeffs = None


class DetectLanes:
    def __init__(self):
        pass

    def generate_frame(self, file, src, dst, filepath=False):
        global prev_left_coeffs
        global prev_right_coeffs

        if filepath:
            # Read in image
            raw = cv2.imread(file)
        else:
            raw = file

        # Parameters
        imshape = raw.shape

        # src = np.float32(
        #     [[120, 720],
        #      [550, 470],
        #      [700, 470],
        #      [1160, 720]])
        #
        # dst = np.float32(
        #     [[200, 720],
        #      [200, 0],
        #      [1080, 0],
        #      [1080, 720]])

        # M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        blank_canvas = np.zeros((720, 1280))
        colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Apply distortion correction to raw image
        cam_calibration = hp.get_camera_calibration()
        cam_calibrator = hp.CalibrateCamera(imshape[:-1], cam_calibration)
        image = cam_calibrator.undistort(raw)

        have_fit = False

        xgrad_thresh_temp = (40, 100)
        s_thresh_temp = (150, 255)

        while have_fit == False:
            # combined_binary = helper.apply_threshold_v2(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)

            perspective_transform = hp.PerspectiveTransformer(src, dst)
            frame = hp.generate_lane_mask(image, 400)
            warped = perspective_transform.transform(frame)

            leftx, lefty, rightx, righty = helper.histogram_pixels(warped, horizontal_offset=40)

            if len(leftx) > 1 and len(rightx) > 1:
                have_fit = True
            xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
            s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

        left_fit, left_coeffs = helper.fit_second_order_poly(lefty, leftx, return_coeffs=True)

        right_fit, right_coeffs = helper.fit_second_order_poly(righty, rightx, return_coeffs=True)

        # # Determine curvature of the lane
        # # Define y-value where we want radius of curvature
        # # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = 500
        left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1]) ** 2) ** 1.5) \
                                    / (2 * left_coeffs[0]))
        right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                                     / (2 * right_coeffs[0]))

        curvature = (left_curverad + right_curverad) / 2
        min_curverad = min(left_curverad, right_curverad)

        if not helper.plausible_curvature(left_curverad, right_curverad) or \
                not helper.plausible_continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs,
                                                            prev_right_coeffs):
            if prev_left_coeffs is not None and prev_right_coeffs is not None:
                left_coeffs = prev_left_coeffs
                right_coeffs = prev_right_coeffs

        prev_left_coeffs = left_coeffs
        prev_right_coeffs = right_coeffs

        # Det vehicle position wrt centre
        centre = helper.center(719, left_coeffs, right_coeffs)

        # 7. Warp the detected lane boundaries back onto the original image.

        polyfit_left = helper.draw_poly(blank_canvas, helper.lane_poly, left_coeffs, 30)
        polyfit_drawn = helper.draw_poly(polyfit_left, helper.lane_poly, right_coeffs, 30)

        # # Convert to colour and highlight lane line area
        trace = colour_canvas
        trace[polyfit_drawn > 1] = [0, 0, 255]

        area = helper.highlight_lane_line_area(blank_canvas, left_coeffs, right_coeffs)
        trace[area == 1] = [0, 255, 0]

        lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

        combined_img = cv2.add(lane_lines, image)
        helper.add_figures_to_image(combined_img, curvature=curvature,
                                    vehicle_position=centre,
                                    min_curvature=min_curverad,
                                    left_coeffs=left_coeffs,
                                    right_coeffs=right_coeffs)

        return combined_img
