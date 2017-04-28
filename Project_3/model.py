import pickle
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.models import Sequential

"""
2. Creating the model
"""

# Read the pickled data.
with open('data.p', mode='rb') as f:
    data = pickle.load(f)

x_train, y_train = data['x_train'], data['y_train']

# Create the model.

