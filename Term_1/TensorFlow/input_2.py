# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    """
    A `placeholder` is like a block which is empty but it has a pre-initialised data type associated to it and the value of it cannot be changed.

    Returns
    -------
        Output: ndarray
    """
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={x: 123})

    return output

if __name__ == '__main__':
    print(run())
