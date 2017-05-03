import preprocessor_modeler
from keras.layers import Flatten, Dense, Lambda, ELU, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

"""
2. Creating the model
"""

# Create the model.
model = Sequential()

# normalise data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64, 64, 3)))

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

model.summary()

history_loss = preprocessor_modeler.LossHistory()

train_gen = preprocessor_modeler.generate_next_batch()
validation_gen = preprocessor_modeler.generate_next_batch()

history = model.fit_generator(train_gen,
                              steps_per_epoch=20032,
                              epochs=8,
                              validation_data=validation_gen,
                              validation_steps=6400,
                              verbose=1, callbacks=[history_loss])

preprocessor_modeler.save_model(history)
preprocessor_modeler.save_loss(history_loss.losses)
