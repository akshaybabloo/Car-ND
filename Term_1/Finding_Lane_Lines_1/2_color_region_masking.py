import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

try:
    image = mpimg.imread('test.jpg')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)
print('This image is: {}, with dimensions: {}'.format(type(image), image.shape))

ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Init a triangle (this part is variable)
left_bottom = [100, 525]  # [0, 539]
right_bottom = [850, 550]  # [900, 539]
apex = [460, 310]  # [475, 320]

# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html for more info
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2])

# See https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html for more info
xx, yy = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (yy > (xx*fit_left[0] + fit_left[1])) & (yy > (xx*fit_right[0] + fit_right[1])) & (yy < (xx*fit_bottom[0] + fit_bottom[1]))

# Mask color selection
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Show figures
f = plt.figure()
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]

f.add_subplot(1, 3, 1)
plt.plot(x, y, 'b--', lw=4)
plt.imshow(image)  # Image superimposed with plotted lines
plt.title("Original image superimposed with plotted lines")

f.add_subplot(1, 3, 2)
plt.imshow(color_select)  # bright color line with triangle
plt.title("Bright lines with triangle selected")

f.add_subplot(1, 3, 3)
plt.imshow(line_image)  # Masked image
plt.title("Original image with masked color")
plt.show()
