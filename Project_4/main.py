import cv2
import matplotlib.pyplot as plt
import pickle

import os
from glob import glob

import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.misc import imread
from tqdm import tqdm


from helper import *

with open("camera_cal/calibration.p", mode='rb') as f:
    camera_calib = pickle.load(f)
mtx = camera_calib["mtx"]
dist = camera_calib["dist"]

prev_left_coeffs = None
prev_right_coeffs = None


def image_pipeline(file, filepath=False):
    global prev_left_coeffs
    global prev_right_coeffs

    plt.clf()

    if filepath == True:
        # Read in image
        raw = cv2.imread(file)
    else:
        raw = file

    # Parameters
    imshape = raw.shape

    src = np.float32(
        [[120, 720],
         [550, 470],
         [700, 470],
         [1160, 720]])

    dst = np.float32(
        [[200, 720],
         [200, 0],
         [1080, 0],
         [1080, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    height = raw.shape[0]
    offset = 50
    offset_height = height - offset
    half_frame = raw.shape[1] // 2
    steps = 6
    pixels_per_step = offset_height / steps
    window_radius = 200
    medianfilt_kernel_size = 51

    blank_canvas = np.zeros((720, 1280))
    colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Apply distortion correction to raw image
    image = cv2.undistort(raw, mtx, dist, None, mtx)

    ## Option I
    combined = apply_thresholds(image)

    ## Option II

    have_fit = False
    curvature_checked = False

    xgrad_thresh_temp = (40, 100)
    s_thresh_temp = (150, 255)

    while have_fit == False:
        combined_binary = apply_threshold_v2(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)
        #    plt.imshow(combined_binary, cmap="gray")

        # Plotting thresholded images
        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Option 1')
        ax1.imshow(combined, cmap="gray")

        ax2.set_title('Option 2: Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        """

        # Warp onto birds-eye-view
        # Previous region-of-interest mask's function is absorbed by the warp
        warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        # plt.imshow(warped, cmap="gray")

        # Histogram and get pixels in window

        leftx, lefty, rightx, righty = histogram_pixels(warped, horizontal_offset=40)

        plt.imshow(warped, cmap="gray")

        if len(leftx) > 1 and len(rightx) > 1:
            have_fit = True
        xgrad_thresh_temp = (xgrad_thresh_temp[0] - 2, xgrad_thresh_temp[1] + 2)
        s_thresh_temp = (s_thresh_temp[0] - 2, s_thresh_temp[1] + 2)

    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
    # print("Left coeffs:", left_coeffs)
    # print("righty[0]: ,", righty[0], ", rightx[0]: ", rightx[0])
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)
    # print("Right coeffs: ", right_coeffs)

    # Plot data
    """
    plt.plot(left_fit, lefty, color='green', linewidth=3)
    plt.plot(right_fit, righty, color='green', linewidth=3)
    plt.imshow(warped, cmap="gray")
    """

    # Determine curvature of the lane
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 500
    left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1]) ** 2) ** 1.5) \
                                / (2 * left_coeffs[0]))
    right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                                 / (2 * right_coeffs[0]))
    # print("Left lane curve radius: ", left_curverad)
    # print("Right lane curve radius: ", right_curverad)
    curvature = (left_curverad + right_curverad) / 2
    min_curverad = min(left_curverad, right_curverad)

    # TODO: if plausible parallel, continue. Else don't make `curvature_checked` = True
    if not plausible_curvature(left_curverad, right_curverad) or \
            not plausible_continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs, prev_right_coeffs):
        if prev_left_coeffs is not None and prev_right_coeffs is not None:
            left_coeffs = prev_left_coeffs
            right_coeffs = prev_right_coeffs

    prev_left_coeffs = left_coeffs
    prev_right_coeffs = right_coeffs

    # Det vehicle position wrt centre
    centre = center(719, left_coeffs, right_coeffs)

    ## 7. Warp the detected lane boundaries back onto the original image.

    # print("Left coeffs: ", left_coeffs)
    # print("Right fit: ", right_coeffs)
    polyfit_left = draw_poly(blank_canvas, lane_poly, left_coeffs, 30)
    polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 30)
    #    plt.imshow(polyfit_drawn, cmap="gray")
    #    plt.imshow(warped)

    # Convert to colour and highlight lane line area
    trace = colour_canvas
    trace[polyfit_drawn > 1] = [0, 0, 255]
    # print("polyfit shape: ", polyfit_drawn.shape)
    area = highlight_lane_line_area(blank_canvas, left_coeffs, right_coeffs)
    trace[area == 1] = [0, 255, 0]
    # plt.imshow(trace)
    lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    # plt.imshow(trace)

    combined_img = cv2.add(lane_lines, image)
    add_figures_to_image(combined_img, curvature=curvature,
                         vehicle_position=centre,
                         min_curvature=min_curverad,
                         left_coeffs=left_coeffs,
                         right_coeffs=right_coeffs)
    plt.imshow(combined_img)
    return combined_img


# def run():
#     """
#     Runs the pipeline.
#     """
#
#     if not os.path.isdir('video' + os.sep + 'seq'):
#         os.mkdir('video' + os.sep + 'seq')
#         to_image_sequence()
#
#     video_in = VideoFileClip(VIDEO_LOCATION)
#     video_size = tuple(video_in.size)  # Get the video frames size.
#
#     # Load calibrated images.
#     cam_calibration = helper.get_camera_calibration()
#     cam_calibrator = helper.CalibrateCamera(video_size, cam_calibration)
#
#     # Load images with img_*.jpeg
#     content = glob('video/seq/img_*.jpeg')
#     images = []
#     for con in tqdm(range(len(content)), desc='Reading files'):
#         # images.append(imread('../video/seq/img_%s.jpeg' % i))
#         images.append(imread('video/seq/img_%s.jpeg' % con))
#
#     # Apply line detection to the read images and write them to a folder.
#     rows = len(images)
#     processed_images = []
#     for row in tqdm(range(rows), desc='Applying DetectLines'):
#         img = images[row]
#
#         ld = processor.DetectLanes(SRC, DST, number_frame=FRAME_MEMORY, camera_calibration=cam_calibrator,
#                                    transform_offset=OFFSET)
#         img = ld.generate_frame(img)
#
#         processed_images.append(img)
#
#         # Write as image
#         im = Image.fromarray(img)
#         im.save('video/seq_new/img_{}.jpeg'.format(row))
#
#     # # Create a backup.
#     # with open('data.p', 'wb') as p:
#     #     pickle.dump({'images': processed_images}, p, protocol=pickle.HIGHEST_PROTOCOL)
#
#     # Read the contents of processed image and make a video of it.
#     new_content = glob('video/seq_new/img_*.jpeg')
#     images_new = []
#
#     # Read the images from the processed folder
#     for i in tqdm(range(len(new_content)), desc='Reading processed images'):
#         images_new.append(imread('video/seq_new/img_%s.jpeg' % i))
#
#     # Write sequence of images to file as a video.
#     new_clip = ImageSequenceClip(images_new, fps=video_in.fps)
#     new_clip.write_videofile('processed_video.mp4')

from moviepy.editor import VideoFileClip


output = 'project_output_colour.mp4'
clip1 = VideoFileClip("video/project_video.mp4")
output_clip = clip1.fl_image(image_pipeline) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)
