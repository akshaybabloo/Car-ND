import tensorflow as tf


def cross_entropy():
    r"""
    Cross Entropy finds the distance between two probability vectors.

    The equation is given

    .. math:: D(S,L)=-\sum_{i}^{n}L_{i}~log(S_{i})


    Returns
    -------

    """
    softmax_data = [0.7, 0.2, 0.1]
    one_hot_data = [1.0, 0.0, 0.0]

    softmax = tf.placeholder(tf.float32)
    one_hot = tf.placeholder(tf.float32)

    cross_entro = -tf.reduce_sum(tf.mul(one_hot, tf.log(softmax)))

    with tf.Session() as sess:
        print(sess.run(cross_entro, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))


if __name__ == '__main__':
    cross_entropy()
