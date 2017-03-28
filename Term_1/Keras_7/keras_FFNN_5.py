"""
A simple two layer Feed Forward Neural Network using Keras.
"""
import math
import pickle

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def run():
    """
    Running the FFNN in Keras.
    
    Make sure ``train.p`` and ``test.p`` is in the same folder when running this program.
    """
    with open('train.p', 'rb') as f:
        data = pickle.load(f)

    with open('./test.p', mode='rb') as f:
        test = pickle.load(f)

    X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], random_state=0, test_size=0.33)

    assert (X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert (X_train.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3."
    assert (X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to the number of labels."
    assert (X_val.shape[1:] == (32, 32, 3)), "The dimensions of the images are not 32 x 32 x 3."

    # Data normalization.
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train = X_train / 255 - 0.5
    X_val = X_val / 255 - 0.5

    # STOP: Do not change the tests below. Your implementation should pass these tests.
    assert (math.isclose(np.min(X_train), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_train), 0.5, abs_tol=1e-5)), \
        "The range of the training data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))
    assert (math.isclose(np.min(X_val), -0.5, abs_tol=1e-5) and math.isclose(np.max(X_val), 0.5, abs_tol=1e-5)), \
        "The range of the validation data is: %.1f to %.1f" % (np.min(X_val), np.max(X_val))

    # Building a two-layer feedforward neural network with Keras.
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(32 * 32 * 3,)))
    model.add(Dense(43, activation='softmax'))

    # STOP: Do not change the tests below. Your implementation should pass these tests.
    dense_layers = []
    for l in model.layers:
        if type(l) == Dense:
            dense_layers.append(l)
    assert (len(dense_layers) == 2), "There should be 2 Dense layers."
    d1 = dense_layers[0]
    d2 = dense_layers[1]
    assert (d1.input_shape == (None, 3072))
    assert (d1.output_shape == (None, 128))
    assert (d2.input_shape == (None, 128))
    assert (d2.output_shape == (None, 43))

    last_layer = model.layers[-1]
    assert (last_layer.activation.__name__ == 'softmax'), "Last layer should be softmax activation, is {}.".format(
        last_layer.activation.__name__)

    # Debugging
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape, l.activation)

    # Compiling and train the model.
    Y_train = np_utils.to_categorical(y_train, 43)
    Y_val = np_utils.to_categorical(y_val, 43)

    X_train_flat = X_train.reshape(-1, 32 * 32 * 3)
    X_val_flat = X_val.reshape(-1, 32 * 32 * 3)

    model.summary()
    # Compiling and train the model.
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X_train_flat, Y_train,
                        batch_size=128, epochs=20,
                        verbose=1, validation_data=(X_val_flat, Y_val))

    # STOP: Do not change the tests below. Your implementation should pass these tests.
    assert (history.history['acc'][-1] > 0.92), "The training accuracy was: %.3f" % history.history['acc'][-1]
    assert (history.history['val_acc'][-1] > 0.9), "The validation accuracy is: %.3f" % history.history['val_acc'][-1]

    # -------------------------------------------------------------------------------------------------------------
    # Testing model
    X_test = test['features']
    y_test = test['labels']
    X_test = X_test.astype('float32')
    X_test /= 255
    X_test -= 0.5
    Y_test = np_utils.to_categorical(y_test, 43)
    X_test_flat = X_test.reshape(-1, 32 * 32 * 3)

    final_test = model.evaluate(X_test_flat, Y_test)
    print()
    print(final_test)


if __name__ == '__main__':
    run()
