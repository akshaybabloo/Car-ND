import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

try:
    image = mpimg.imread('exit-ramp.jpg')
except FileNotFoundError as e:
    print(e)

# Converts an image into 8-bit gray scale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 8-bit image

# Adds a blur to the image see http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
kernel_size = 5  # should be odd numbers
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Canny edge detection see http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
low_threshold = 50
high_threshold = 110
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


# show images
f = plt.figure()

f.add_subplot(1, 3, 1)
plt.imshow(image)
f.add_subplot(1, 3, 2)
plt.imshow(blur_gray, cmap='gray')
f.add_subplot(1, 3, 3)
plt.imshow(edges, cmap='Greys_r')
plt.show()
