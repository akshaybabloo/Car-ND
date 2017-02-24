import tensorflow as tf


def python_run():
    """
    Pythonic way of getting the ``new width`` and ``new height``
    """
    input_shape = [32, 32, 3]  # HxWxD
    filter_shape = [20, 8, 8, 3]  # number_of_filtersxHxWxD
    stride = 2  # S
    valid_padding = 1  # P

    new_height = (input_shape[0] - filter_shape[1] + 2 * valid_padding) / stride + 1
    new_width = (input_shape[1] - filter_shape[2] + 2 * valid_padding) / stride + 1
    new_depth = filter_shape[0]  # number of filters is the depth

    print("{}x{}x{}".format(new_height, new_width, new_depth))


def tensor_run():
    """
    Running convolution in 2D with TensorFlow.
    """
    inputs = tf.placeholder(tf.float32, (None, 32, 32, 3))
    filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20)))  # (height, width, input_depth, output_depth)
    filter_bias = tf.Variable(tf.zeros(20))
    strides = [1, 2, 2, 1]  # (batch, height, width, depth)
    padding = 'VALID'
    conv = tf.nn.conv2d(inputs, filter_weights, strides, padding) + filter_bias

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(conv, feed_dict={})  # Exception will occur, not a complete code.


if __name__ == '__main__':
    python_run()
    tensor_run()
