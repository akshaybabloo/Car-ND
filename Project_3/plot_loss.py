"""
3. Plotting the loss values from the saved file
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

loss_file = os.path.abspath('model_loss.csv')
loss_data = pd.read_csv(loss_file, sep=',', header=None)

ax = loss_data.plot(kind='line', colormap='jet', markersize=10, title="Loss of 100 Epochs for training the data")
ax.legend(["Loss"])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Value")

avg_loss_data = loss_data.groupby(loss_data.index // 1025).mean()
ax2 = avg_loss_data.plot(kind='line', colormap='jet', markersize=10,
                         title="Average Loss 100 Epochs for training the data")
ax2.legend(["Loss"])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss Value")

plt.show()
