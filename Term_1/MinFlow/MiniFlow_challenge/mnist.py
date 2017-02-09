import os
import struct

import numpy as np
import matplotlib.pyplot as plt
import pandas

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    fname_img_training = os.path.join(path, 'train-images.idx3-ubyte')
    fname_lbl_training = os.path.join(path, 'train-labels.idx1-ubyte')

    fname_img_testing = os.path.join(path, 't10k-images.idx3-ubyte')
    fname_lbl_testing = os.path.join(path, 't10k-labels.idx1-ubyte')

    # Load everything in some numpy arrays for training
    with open(fname_lbl_training, 'rb') as flbl_training:
        magic, num = struct.unpack(">II", flbl_training.read(8))
        lbl_training = np.fromfile(flbl_training, dtype=np.int8)

    with open(fname_img_training, 'rb') as fimg_training:
        magic, num, rows, cols = struct.unpack(">IIII", fimg_training.read(16))
        img_training = np.fromfile(fimg_training, dtype=np.uint8).reshape(len(lbl_training), rows, cols)

    # ----------------------------------------------------------------------------------------------------------

    # Load everything in some numpy arrays for testing
    with open(fname_lbl_testing, 'rb') as flbl_testing:
        magic, num = struct.unpack(">II", flbl_testing.read(8))
        lbl_testing = np.fromfile(flbl_testing, dtype=np.int8)

    with open(fname_img_testing, 'rb') as fimg_testing:
        magic, num, rows, cols = struct.unpack(">IIII", fimg_testing.read(16))
        img_testing = np.fromfile(fimg_testing, dtype=np.uint8).reshape(len(lbl_testing), rows, cols)

    # -----------------------------------------------------------------------------------------------------------

    images = {'training': img_training, 'testing': img_testing}

    return images


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap='gray')
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

if __name__ == '__main__':
    img = read()

    for i in img['training']:
        show(i)
        break

    for i in img['testing']:
        show(i)
        break
