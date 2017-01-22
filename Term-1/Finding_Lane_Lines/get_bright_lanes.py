import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# np.set_printoptions(threshold=np.nan)

try:
    image = mpimg.imread('test.jpg')  # Reading a image file.
except FileNotFoundError as e:
    print(e)
print('This image is: {}, with dimensions: {}'.format(type(image), image.shape))

# Making a copy of the image.
color_select = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Select any pixels less than given threshold.
thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2])
color_select[thresholds] = [0, 0, 0]

# Show image.
plt.imshow(color_select)
plt.show()
