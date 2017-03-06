"""
Set the values to `strides` and `ksize` such that
the output shape after pooling is (1, 2, 2, 1).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.max_pool` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def maxpool(input):
    """
    Max Pool using TensorFlow
    Parameters
    ----------
    input

    Returns
    -------
    tensor
    """
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 3, 4]
    padding = 'SAME'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#max_pool
    return tf.nn.max_pool(input, ksize, strides, padding)


def run():
    """
    Run TensorFlow.
    """

    out = maxpool(X)
    print(out)

if __name__ == '__main__':
    run()
