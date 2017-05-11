"""
Script to run all the methods & function and view them appropriately.
"""

import helper
import processor
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import ImageSequenceClip
from scipy.misc import imread
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import pickle

HIST_STEPS = 10
OFFSET = 250
FRAME_MEMORY = 5
SRC = np.float32([
    (132, 703),
    (540, 466),
    (740, 466),
    (1147, 703)])

DST = np.float32([
    (SRC[0][0] + OFFSET, 720),
    (SRC[0][0] + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, 720)])


VIDEO_LOCATION = 'video/project_video.mp4'


def to_image_sequence():
    """
    Convert video to image.
    """
    video_in = VideoFileClip(VIDEO_LOCATION)

    for idx, frame in enumerate(video_in.iter_frames()):
        im = Image.fromarray(frame)
        im.save('video/seq/img_{}.jpeg'.format(idx))


def run():
    """
    
    """

    video_in = VideoFileClip(VIDEO_LOCATION)
    video_size = tuple(video_in.size)

    cam_calibration = helper.get_camera_calibration()

    cam_calibrator = helper.CalibrateCamera(video_size, cam_calibration)

    content = glob('video/seq/img_*.jpeg')

    images = []

    for i in range(len(content)):
        # images.append(imread('../video/seq/img_%s.jpeg' % i))
        images.append(imread('video/seq/img_%s.jpeg' % i))
        print(len(content)-i)

    rows = len(images)
    # add_to_me = []
    for row in range(rows):
        img = images[row]

        ld = processor.DetectLanes(SRC, DST, number_frame=FRAME_MEMORY, camera_calibration=cam_calibrator, transform_offset=OFFSET)
        img = ld.generate_frame(img)

        # add_to_me.append(img)

        im = Image.fromarray(img)
        im.save('video/seq_new/img_{}.jpeg'.format(row))
        print(rows-row)


    # with open('data.p', 'w'):
    #     pickle.dumps({'images': add_to_me}, protocol=pickle.HIGHEST_PROTOCOL)
    # print(len(add_to_me))

if __name__ == '__main__':
    run()
    # to_image_sequence()
