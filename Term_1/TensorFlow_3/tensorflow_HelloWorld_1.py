import tensorflow as tf


def run():
    """
    Printing `Hello World` using Tensor `constant`.
    """
    # Create TensorFlow object called tensor
    hello_constant = tf.constant('Hello World!')

    with tf.Session() as sess:
        # Run the tf.constant operation in the session
        output = sess.run(hello_constant)
        print(output)

if __name__ == '__main__':
    run()
