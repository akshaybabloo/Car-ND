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
SRC = np.float32(
    [[120, 720],
     [550, 470],
     [700, 470],
     [1160, 720]])

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
    Runs the pipeline.
    """

    if not os.path.isdir('video' + os.sep + 'seq'):
        os.mkdir('video' + os.sep + 'seq')
        to_image_sequence()

    video_in = VideoFileClip(VIDEO_LOCATION)

    # Load images with img_*.jpeg
    content = glob('video/seq/img_*.jpeg')
    images = []
    for con in tqdm(range(len(content)), desc='Counting files'):
        # images.append(imread('video/seq/img_%s.jpeg' % i))
        images.append('video/seq/img_%s.jpeg' % con)

    # Apply line detection to the read images and write them to a folder.
    ld = processor.DetectLanes()
    rows = len(images)
    processed_images = []
    for row in tqdm(range(rows), desc='Applying DetectLines'):

        img = ld.generate_frame('video/seq/img_%s.jpeg' % row, filepath=True)

        processed_images.append(img)

        # Write as image
        im = Image.fromarray(img)
        im.save('video/seq_new/img_{}.jpeg'.format(row))

    # # Create a backup.
    # with open('data.p', 'wb') as p:
    #     pickle.dump({'images': processed_images}, p, protocol=pickle.HIGHEST_PROTOCOL)

    # Read the contents of processed image and make a video of it.
    new_content = glob('video/seq_new/img_*.jpeg')
    images_new = []

    # Read the images from the processed folder
    for i in tqdm(range(len(new_content)), desc='Reading processed images'):
        images_new.append(imread('video/seq_new/img_%s.jpeg' % i))

    # Write sequence of images to file as a video.
    new_clip = ImageSequenceClip(images_new, fps=video_in.fps)
    new_clip.write_videofile('processed_video.mp4')


if __name__ == '__main__':
    run()
