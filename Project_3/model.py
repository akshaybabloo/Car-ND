import os
import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import preprocessor_modeler

"""
2. Creating the model
"""


# Read the pickled data.
# with open('data.p', mode='rb') as f:
#     data = pickle.load(f)
#
# x_train, y_train = data['x_train'], data['y_train']

# Create the model.
model = Sequential()

# normalise data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))

# model.add(Cropping2D(cropping=((70, 25), (0, 0))))

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

# check_point = ModelCheckpoint(filepath="temp/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)

history_loss = preprocessor_modeler.LossHistory()
# model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=100, callbacks=[history, check_point])

train_gen = preprocessor_modeler.generate_next_batch()
validation_gen = preprocessor_modeler.generate_next_batch()

history = model.fit_generator(train_gen,
                              steps_per_epoch =20032,
                              epochs=8,
                              validation_data=validation_gen,
                              validation_steps=6400,
                              verbose=1, callbacks=[history_loss])

preprocessor_modeler(history)

# Save loss to file
loss_array = np.asarray(history.losses)
np.savetxt("model_loss.csv", loss_array, fmt='%10.5f', delimiter=",")

# # Saving model to json file
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
#
# # Saving weights
# model.save_weights("model_weights.h5")
# print("Saved weights to disk")
#
# # Saving model
# model.save("model.h5")
# print("Saved model to disk")
