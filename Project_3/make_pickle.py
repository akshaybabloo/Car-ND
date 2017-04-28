import csv
import os
import pickle
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("Pickler started.")
log.info("Reading path.")

full_path_csv = os.path.abspath('data' + os.sep + 'driving_log.csv')
full_path_img = os.path.abspath('data' + os.sep + 'IMG') + os.sep

"""
1. Preparing the data.
"""
log.info("Reading driving_log.csv")
# Reading the driving_log.csv
lines = []
with open(full_path_csv) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

log.info("driving_log.csv read.")
log.info('Reading images.')

# Reading the images (center, left, right) and its measurements
images = []
measurements = []
for line in lines:
    for camera in range(3):
        source_path = line[camera]
        file_name = source_path.split('/')[-1]
        # current_path = os.path.abspath(full_path_img + file_name)
        image = cv2.imread(file_name)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

log.info('Images read.')
log.info('Augmenting data.')

# Augmenting the image data and its measurement. Flipping them vertically.
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))  # Flipping them vertically
    augmented_measurements.append(measurement * -1.0)

log.info('Data augmented.')
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

log.info('Creating pickle.')
data = {'x_train': x_train, 'y_train': y_train}

with open('data.p', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

log.info("Data pickled.")
