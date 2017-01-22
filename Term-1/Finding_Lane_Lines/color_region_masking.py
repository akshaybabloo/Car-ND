import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

np.set_printoptions(threshold=np.nan)

try:
    image = mpimg.imread('test.jpg')
except FileNotFoundError as e:
    print(e)
print('This image is: {}, with dimensions: {}'.format(type(image), image.shape))

ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2])

xx, yy = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (yy > (xx*fit_left[0] + fit_left[1])) & (yy > (xx*fit_right[0] + fit_right[1])) & (yy < (xx*fit_bottom[0] + fit_bottom[1]))

color_select[color_thresholds] = [0, 0, 0]

line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

plt.imshow(color_select)
plt.imshow(line_image)
plt.show()
