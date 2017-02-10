import tensorflow as tf


def run():
    """
    Setting a variable in TensorFlow
    """
    var = tf.Variable([10.0, 20.0])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        b = sess.run(var)
        print(b)

if __name__ == '__main__':
    run()
