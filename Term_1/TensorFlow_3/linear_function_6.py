# Solution is available in the other "sandbox_solution.py" tab
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Solution is available in the other "quiz_solution.py" tab
def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights

    Parameters
    ----------

    n_features: Number of features
    n_labels: Number of labels

    Returns
    -------

    TensorFlow weights: ndarray
    """
    # TODO: Return weights
    tf.Variable(tf.truncated_normal((n_features, n_labels)))


def get_biases(n_labels):
    """
    Return TensorFlow bias

    Parameters
    ----------

    n_labels: Number of labels

    Returns
    -------

    TensorFlow bias: ndarray
    """
    # TODO: Return biases
    tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in TensorFlow

    Parameters
    ----------

    input: TensorFlow input
    w: TensorFlow weights
    b: TensorFlow biases

    Returns
    -------

    TensorFlow linear function: ndarray
    """
    # TODO: Linear Function (xW + b)
    tf.add(tf.mul(input, w), b)


def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset

    Parameters
    ----------

    n_labels: Number of labels to use

    Returns
    -------

    Tuple of feature list and label list: ndarray, ndarray
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets('.\MNIST_data', one_hot=True)

    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels


if __name__ == '__main__':

    init = tf.global_variables_initializer()
    # Number of features (28*28 image is 784 features)
    n_features = 784
    # Number of labels
    n_labels = 3

    # Features and Labels
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    # Weights and Biases
    w = get_weights(n_features, n_labels)
    b = get_biases(n_labels)

    # Linear Function xW + b
    logits = linear(features, w, b)

    # Training data
    train_features, train_labels = mnist_features_labels(n_labels)

    with tf.Session() as session:
        session.run(init)

        # Softmax
        prediction = tf.nn.softmax(logits)

        # Cross entropy
        # This quantifies how far off the predictions were.
        # You'll learn more about this in future lessons.
        cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

        # Training loss
        # You'll learn more about this in future lessons.
        loss = tf.reduce_mean(cross_entropy)

        # Rate at which the weights are changed
        # You'll learn more about this in future lessons.
        learning_rate = 0.08

        # Gradient Descent
        # This is the method used to train the model
        # You'll learn more about this in future lessons.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Run optimizer and get loss
        _, l = session.run(
            [optimizer, loss],
            feed_dict={features: train_features, labels: train_labels})

    # Print loss
    print('Loss: {}'.format(l))
