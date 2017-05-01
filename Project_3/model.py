import os
import pickle

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

"""
2. Creating the model
"""


class LossHistory(Callback):
    """
    Getting the loss for the epochs.
    """

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))


# Read the pickled data.
with open('data.p', mode='rb') as f:
    data = pickle.load(f)

x_train, y_train = data['x_train'], data['y_train']

# Create the model.
model = Sequential()

# normalise data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Conv layer 1
model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
model.add(ELU())

# Conv layer 2
model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

# Conv layer 3
model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))

# Flatten the data
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())

# Fully connected layer 1
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())

# Fully connected layer 2
model.add(Dense(50))
model.add(ELU())

model.add(Dense(1))

adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

print("Model summary:\n", model.summary())

# Create temp directory if it does not exist.
if not os.path.isdir('temp'):
    os.mkdir('temp')

check_point = ModelCheckpoint(filepath="temp/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)

history = LossHistory()
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=100, callbacks=[history, check_point])

# Save loss to file
loss_array = np.asarray(history.losses)
np.savetxt("model_loss.csv", loss_array, fmt='%10.5f', delimiter=",")

# Saving model to json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Saving weights
model.save_weights("model_weights.h5")
print("Saved weights to disk")

# Saving model
model.save("model.h5")
print("Saved model to disk")
