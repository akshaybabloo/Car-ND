import os
import pickle

from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Convolution2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

"""
2. Creating the model
"""

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
model.add(Convolution2D(16, (8, 8), subsample=(4, 4), border_mode="same"))
model.add(ELU())

# Conv layer 2
model.add(Convolution2D(32, (5, 5), subsample=(2, 2), border_mode="same"))
model.add(ELU())

# Conv layer 3
model.add(Convolution2D(64, (5, 5), subsample=(2, 2), border_mode="same"))

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

model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=100, callbacks=[check_point])

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
