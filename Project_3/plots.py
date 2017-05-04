"""
3. Plotting the loss values from the saved file
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import preprocessor_modeler

loss_file = os.path.abspath('model_loss.csv')
loss_data = pd.read_csv(loss_file, sep=',', header=None)

f = plt.figure()

f.add_subplot(3, 1, 1)
ax = loss_data.plot(kind='line', colormap='jet', markersize=10, title="Loss of 100 Epochs for training the data")
ax.legend(["Loss"])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Value")

f.add_subplot(3, 1, 2)
avg_loss_data = loss_data.groupby(loss_data.index // 20032).mean()
ax2 = avg_loss_data.plot(kind='line', colormap='jet', markersize=10,
                         title="Average Loss 100 Epochs for training the data")
ax2.legend(["Loss"])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss Value")

# Comment below lines to plot the abouve two correctly.
y_data = None
for content in preprocessor_modeler.generate_next_batch():
    y_data = content[1]
    break

f.add_subplot(3, 1, 3)
sort_data = np.sort(y_data)
fit = stats.norm.pdf(sort_data, np.mean(sort_data), np.std(sort_data))
plt.plot(sort_data, fit, '-o')
plt.hist(sort_data, normed=True)

plt.show()
