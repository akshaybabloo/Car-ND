import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
from keras.callbacks import Callback

# Some useful constants
DRIVING_LOG_FILE = os.path.abspath('data' + os.sep + 'driving_log.csv')
IMG_PATH = os.path.abspath('data' + os.sep + 'IMG') + os.sep
STEERING_COEFFICIENT = 0.229


def crop(image, top_percent, bottom_percent):
    """
    Image cropped based on the input percentage.
    
    Parameters
    ----------
    image: ndarray
        Image
    top_percent: float
        Percentage ranging from ``0`` to ``1``
    bottom_percent: float
        Percentage ranging from ``0`` to ``1``

    Returns
    -------
    image: ndarray
        Cropped image.
    """

    assert 0.0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0.0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def random_flip(image, steering_angle, flipping_prob=0.5):
    """
    Using bernoulli's discreet random variable (mimicking random coin flip) return the image and the steering angle.
    
    Parameters
    ----------
    image: ndarray
        Image
    steering_angle: float
        Steering angle.
    flipping_prob: float
        Flipping probability.

    Returns
    -------
    image, steering_angle: tuple
        Flipped image and negated steering angle.
    """
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    """
    Adding random brightness (gamma) to the data - http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    
    Parameters
    ----------
    image: Numpy array
        Source image.

    Returns
    -------
    Numpy array: Numpy array
        Gamma corrected image.

    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    """
    Adding random noise to the data - https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    
    Parameters
    ----------
    image: ndarray
        Image.
    steering_angle: float
        Steering angle.
    shear_range:
        Random range of given number.

    Returns
    -------
    image, steering_angle: tuple
        Random sheered image and steering angle
    """

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def random_rotation(image, steering_angle, rotation_amount=15):
    """
    Adding random rotations to the car.
    
    Parameters
    ----------
    image: ndarray
        Image.
    steering_angle: float
        Steering angle
    rotation_amount: int
        Add or subtract this number to ``steering_angle``

    Returns
    -------
    Rotated image: ndarray
        Rotated image
    """

    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad


def min_max(data, a=-0.5, b=0.5):
    """
    Finding the minimum and maximum of a given ndarray or list. 
    
    Parameters
    ----------
    data: ndarray or list
        Data.
    a: float
        Minimum number 
    b: float
        Maximum number

    Returns
    -------
    array: ndarray or list
        ndarray or list of minimum and maximum numbers.
    """

    data_max = np.max(data)
    data_min = np.min(data)
    return a + (b - a) * ((data - data_min) / (data_max - data_min))


def resize(image, new_dim):
    """
    Resize image according to the new dimensions.
    
    Parameters
    ----------
    image: ndarray
        Image.
    new_dim: tuple
        A tuple of new dimensions.

    Returns
    -------
    Resize image: ndarray
        A numpy array or resized image.

    """

    return scipy.misc.imresize(image, new_dim)


def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    """
    Generate a new image after applying random shear, cropping, random flip, ransom gamma and resize.
    
    Parameters
    ----------
    image: ndarray
        Image.
    steering_angle: float
        Angle of steering, could be in ``+`` or ``-``
    top_crop_percent: float
        Split percentage between ``0`` and ``1``
    bottom_crop_percent: float
        Cropping percentage between ``0`` and ``1``
    resize_dim: tuple
        A tuple of resizing dimensions
    do_shear_prob: float
        Between ``0`` and ``1``

    Returns
    -------
    image, angle: tuple
        Processed image and angle.

    """

    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle


def get_next_image_files(batch_size=64):
    """
    The simulator records three images (namely: left, center, and right) at a given time
    However, when we are picking images for training we randomly (with equal probability)
    one of these three images and its steering angle.
    
    Parameters
    ----------
    batch_size: int
        Size of the image batch

    Returns
    -------
    image_files_and_angles: list
        A list of selected (image files names, respective steering angles)

    """
    data = pd.read_csv(DRIVING_LOG_FILE)
    data.columns = ['center', 'left', 'right', 'left_angle', 'acceleration', 'right_angle', 'speed']
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            img = img.split(os.sep)[-1]
            angle = data.iloc[index]['left_angle'] + STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            img = img.split(os.sep)[-1]
            angle = data.iloc[index]['left_angle']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            img = img.split(os.sep)[-1]
            angle = data.iloc[index]['left_angle'] - STEERING_COEFFICIENT
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_next_batch(batch_size=64):
    """
    This generator yields the next training batch
    
    Parameters
    ----------
    batch_size: int
        Number of training images in a single batch

    Returns
    -------
    ndarray: tuple
        A tuple of features and steering angles as two numpy arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(IMG_PATH + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)


def save_loss(loss, file_name="model_loss.csv"):
    """
    Save model's loss to csv.
    
    Parameters
    ----------
    loss: list
        List of errors.
    file_name: str
        File name.
    """

    loss_array = np.asarray(loss)
    np.savetxt(file_name, loss_array, fmt='%10.5f', delimiter=",")


class LossHistory(Callback):
    """
    Getting the loss for the epochs.
    """

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
