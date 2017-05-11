"""
Script to run all the methods & function and view them appropriately.
"""

import os
import pickle
from glob import glob

import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.misc import imread
from tqdm import tqdm

import helper
import processor

OFFSET = 250
FRAME_MEMORY = 1
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

    for idx, frame in tqdm(enumerate(video_in.iter_frames()), desc='Converting video to images'):
        im = Image.fromarray(frame)
        im.save('video' + os.sep + 'seq' + os.sep + 'img_{}.jpeg'.format(idx))


def run():
    """
    Runs the flow.
    """

    if not os.path.isdir('video' + os.sep + 'seq'):
        os.mkdir('video' + os.sep + 'seq')
        to_image_sequence()

    video_in = VideoFileClip(VIDEO_LOCATION)
    video_size = tuple(video_in.size)

    cam_calibration = helper.get_camera_calibration()

    cam_calibrator = helper.CalibrateCamera(video_size, cam_calibration)

    content = glob('video/seq/img_*.jpeg')
    images = []

    for con in tqdm(range(len(content)), desc='Reading files'):
        # images.append(imread('../video/seq/img_%s.jpeg' % i))
        images.append(imread('video/seq/img_%s.jpeg' % con))

    rows = len(images)
    processed_images = []
    for row in tqdm(range(rows), desc='Applying DetectLines'):
        img = images[row]

        ld = processor.DetectLanes(SRC, DST, number_frame=FRAME_MEMORY, camera_calibration=cam_calibrator,
                                   transform_offset=OFFSET)
        img = ld.generate_frame(img)

        processed_images.append(img)

        # Write as image
        im = Image.fromarray(img)
        im.save('video/seq_new/img_{}.jpeg'.format(row))

    # Create a backup.
    with open('data.p', 'w') as p:
        pickle.dump({'images': processed_images}, p, protocol=pickle.HIGHEST_PROTOCOL)

    # Read the contents of processed image and make a video of it.
    new_content = glob('video/seq_new/img_*.jpeg')
    images_new = []

    for i in range(len(new_content)):
        # images.append(imread('../video/seq/img_%s.jpeg' % i))
        images_new.append(imread('video/seq_new/img_%s.jpeg' % i))

    # Write sequence of images to file.
    new_clip = ImageSequenceClip(images_new, fps=video_in.fps)
    new_clip.write_videofile('processed_video.mp4')


if __name__ == '__main__':
    run()
