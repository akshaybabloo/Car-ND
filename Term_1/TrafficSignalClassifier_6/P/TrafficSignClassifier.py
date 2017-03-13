import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

TRAINING_FILE = 'train.p'
VALIDATION_FILE = 'valid.p'
TESTING_FILE = 'test.p'

with open(TRAINING_FILE, mode='rb') as f:
    train = pickle.load(f)
with open(VALIDATION_FILE, mode='rb') as f:
    valid = pickle.load(f)
with open(TESTING_FILE, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Normalize data
X_train = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
X_validation = (X_validation - X_validation.mean()) / (np.max(X_validation) - np.min(X_validation))
X_test = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))


def get_description():
    """
    Prints the description of the dataset.
    """
    n_train = len(X_train)
    n_test = len(X_test)

    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    image_shape = np.shape(image)

    n_classes = len(y_train) + len(y_test)

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)


def show_image():
    """
    Displays a random images from the dataset with the class label.
    """
    num = 1
    index = random.sample(range(len(X_train)), 4)

    fig = plt.figure()

    for n in index:
        fig.add_subplot(2, 2, num)
        image = X_train[n].squeeze()
        plt.imshow(image)
        plt.title("Class Label {}".format(n))
        num += 1

    fig.tight_layout()
    plt.show()


X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 100
BATCH_SIZE = 100


def LeNet(x):
    """
    CNN for the dataset.
    Parameters
    ----------
    x

    Returns
    -------
    tensor
    """
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    MU = 0
    SIGMA = 0.1
    PADDING = 'VALID'

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=MU, stddev=SIGMA))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding=PADDING) + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=PADDING)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=MU, stddev=SIGMA))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding=PADDING) + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=MU, stddev=SIGMA))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=MU, stddev=SIGMA))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=MU, stddev=SIGMA))
    fc3_b = tf.Variable(tf.zeros(43))
    _logits = tf.matmul(fc2, fc3_w) + fc3_b

    return _logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

RATE = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=RATE)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    """
    Evaluates the total accuracy for the given inputs.
    Parameters
    ----------
    X_data
    y_data

    Returns
    -------
    numpy.float64

    """
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def run_training():
    """
    Run the training and print the accuracy.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        validation_accuracy = 0

        print("Training...")
        print()
        for i in range(EPOCHS):
            if validation_accuracy >= 0.95:
                print('Break')
                break
            _X_train, _y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = _X_train[offset:end], _y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")


def run_testing():
    """
    Run testing with new data and get the accuracy.
    """
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


if __name__ == '__main__':
    run_training()
    run_testing()
    # show_image()
