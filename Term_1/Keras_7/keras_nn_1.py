"""
A simple  Multi-Layer Perceptron model using Keras framework. Using the German signals dataset - http://benchmark.ini.rub.de
"""

import pickle

import numpy as np
import tensorflow as tf
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer


def run():
    """
    Running the MLP model.

    Returns
    -------

    """
    tf.python.control_flow_ops = tf

    with open('train.p', mode='rb') as f:
        data = pickle.load(f)

    X_train, y_train = data['features'], data['labels']

    # Sequential model
    model = Sequential()

    # 1st Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))

    # 2nd Layer - Add a fully connected layer
    model.add(Dense(128))

    # 3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    # 4th Layer - Add a fully connected layer
    model.add(Dense(5))

    # 5th Layer - Add a ReLU activation layer
    model.add(Activation('softmax'))

    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5)

    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)


if __name__ == '__main__':
    run()
