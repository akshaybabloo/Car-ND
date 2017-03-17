"""
Max pooling in Keras.
"""

import pickle
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.preprocessing import LabelBinarizer


def run():
    tf.python.control_flow_ops = tf

    with open('train.p', mode='rb') as f:
        data = pickle.load(f)

    X_train, y_train = data['features'], data['labels']

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # Preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5)

    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)


if __name__ == '__main__':
    run()
