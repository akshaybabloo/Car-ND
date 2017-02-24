import tensorflow as tf


def run():
    """
    Generating random numbers from a normal distribution.

    The magnitude is no more than 2 standard deviation from the mean.

    `tf.zeros()` returns a ndarray of zeros.
    """
    features = 100
    lables = 5
    weights = tf.Variable(tf.truncated_normal((features, lables)))
    bias = tf.Variable(tf.zeros(lables))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        weights_ = sess.run(weights)
        bias_ = sess.run(bias)
        print(weights_)
        print(bias_)

if __name__ == '__main__':
    run()
