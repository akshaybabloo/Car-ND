import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys

try:
    image = mpimg.imread('exit-ramp.jpg')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 8-bit image

kernel_size = 5  # should be odd numbers
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 110
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


# Parameters for Hough lines
rho = 1
theta = np.pi / 180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image) * 0  # create a blank to draw the lines
# See http://docs.opencv.org/3.2.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate through the lines and draw a line
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Color binary image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

# show images
f = plt.figure()

f.add_subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original image')

f.add_subplot(2, 2, 2)
plt.imshow(blur_gray, cmap='gray')
plt.title('Gaussian blurred gray image')

f.add_subplot(2, 2, 3)
plt.imshow(edges, cmap='Greys_r')
plt.title("Canny edges of Gaussian image")

f.add_subplot(2, 2, 4)
plt.imshow(combo)
plt.title("Hough transformed image of Canny edges")
plt.show()
