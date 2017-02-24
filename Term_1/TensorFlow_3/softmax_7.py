import numpy as np
import tensorflow as tf


def softmax(x):
    r"""
    The equation is given by

    .. math::

        S(y_{i})=\frac{e^{y_{i}}}{\sum_{j}^{n}e^{y_{j}}}

    Parameters
    ----------
    x

    Returns
    -------

    """
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=0))


def tensor_softmax(logit_data):
    """
    Finding softmax using TensorFlow

    Parameters
    ----------
    logit_data

    Returns
    -------

    TensorFlow Softmax: ndarray

    """
    output = None
    logits = tf.placeholder(tf.float32)

    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output

if __name__ == '__main__':

    logits = [3.0, 1.0, 0.2]

    print(softmax(logits))
    print(tensor_softmax(logits))
