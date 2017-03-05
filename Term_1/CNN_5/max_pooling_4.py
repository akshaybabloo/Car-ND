import tensorflow as tf


def run_python():
    """
    Getting the new height and new width for max pooling
    """
    input_height = 4
    input_width = 4
    input_depth = 5
    filter_height = 2
    filter_width = 2
    s = 2  # stride

    new_height = (input_height - filter_height)/s + 1
    new_width = (input_width - filter_width)/s + 1

    print("new height = {}, new width = {} and depth is {}".format(new_height, new_width, input_depth))


def run_tf():
    """
    Getting the new height and new width for max pooling using TensorFlow
    """
    place_holder = tf.placeholder(tf.float32, (None, 4, 4, 5))
    filter_shape = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    pool = tf.nn.max_pool(place_holder, filter_shape, strides, padding)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(pool, feed_dict={})  # Exception might occur.
        print(output)

if __name__ == '__main__':
    run_python()
    run_tf()
