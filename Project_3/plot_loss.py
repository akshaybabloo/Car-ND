"""
3. Plotting the loss values from the saved file
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

loss_file = os.path.abspath('model_loss.csv')

loss_data = pd.read_csv(loss_file, sep=',', header=None)
loss_data.plot(kind='line')
plt.show()
