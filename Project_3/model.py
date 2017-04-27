import csv
import os

import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.models import Sequential

full_path_csv = os.path.abspath('data' + os.sep + 'driving_log.csv')
full_path_img = os.path.abspath('data' + os.sep + 'IMG') + os.sep

# Reading the contents in the CSV file
lines = []
with open(full_path_csv) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

# Reading the images (center, left, right) and its measurements
images = []
measurements = []
for line in lines:
    for camera in range(3):
        source_path = line[camera]
        file_name = source_path.split('/')[-1]
        current_path = full_path_img + file_name
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

# Augmenting the image data and its measurement. Flipping them vertically.
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))  # Flipping them vertically
    augmented_measurements.append(measurement * -1.0)

x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))  # normalise data
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
