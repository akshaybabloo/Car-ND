import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
import sys


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_custom(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, alpha=0.8, beta=1., lamb=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    lamb is lambda
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lamb)


def draw_lines_custom(img, lines, color=[255, 0, 0], thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # Initialise arrays
    positive_slope_points = []
    negative_slope_points = []
    positive_slope_intercept = []
    negative_slope_intercept = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y1 - y2) / (x1 - x2)
            # print("Points: ", [x1, y1, x2, y2])
            length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # print("Length: ", length)
            if not math.isnan(slope):
                if length > 50:
                    if slope > 0:
                        positive_slope_points.append([x1, y1])
                        positive_slope_points.append([x2, y2])
                        positive_slope_intercept.append([slope, y1 - slope * x1])
                    elif slope < 0:
                        negative_slope_points.append([x1, y1])
                        negative_slope_points.append([x2, y2])
                        negative_slope_intercept.append([slope, y1 - slope * x1])

    # Get intercept and coefficient of fitted lines
    pos_coef, pos_intercept = find_line_fit(positive_slope_intercept)
    neg_coef, neg_intercept = find_line_fit(negative_slope_intercept)

    # Get intersection point
    intersection_x_coord = intersection_x(pos_coef, pos_intercept, neg_coef, neg_intercept)

    # Plot lines
    draw_sep_lines(pos_coef, pos_intercept, intersection_x_coord, img, color, thickness)
    draw_sep_lines(neg_coef, neg_intercept, intersection_x_coord, img, color, thickness)


def intersection_x(coef1, intercept1, coef2, intercept2):
    """Returns x-coordinate of intersection of two lines."""
    x = (intercept2 - intercept1) / (coef1 - coef2)
    return x


def draw_sep_lines(coef, intercept, intersection_x, img, color, thickness):

    imshape = img.shape
    # Get starting and ending points of regression line, ints.
    # print("Coef: ", coef, "Intercept: ", intercept,
    #       "intersection_x: ", intersection_x)
    point_one = (int(intersection_x), int(intersection_x * coef + intercept))
    point_two = 0
    if coef > 0:
        point_two = (imshape[1]-1, int(imshape[1] * coef + intercept))
    elif coef < 0:
        point_two = (0, int(0 * coef + intercept))
    print("Point one: ", point_one, "Point two: ", point_two)

    test = new_coordinates(point_one, point_two)
    # test2 = (482,508)
    # test2 = new_coordinates((400,400), (600,200))

    # Draw line using cv2.line
    cv2.line(img, test, point_two, color, thickness)
    # cv2.line(img, (400,400), (600,200), [0,255,0], thickness)
    # cv2.line(img, (400,400), test2, [204,255,204], thickness)
    # cv2.line(img, point_one, point_two, color, thickness)
    print("--------------------------------------------------------------------------------------------------------")


# def new_coordinates(point_one, point_two):
#     print(("x1", point_one[0], "y1", point_one[1]), ("x2", point_two[0], "y2", point_two[1]))
#     distance = math.sqrt((point_two[0] - point_one[0]) ** 2 + (point_two[1] - point_one[1]) ** 2)
#     print("distance between point one and two", distance)
#
#     slope = (point_two[1] - point_one[1]) / (point_two[0] - point_one[0])
#     print("Slope", slope)
#
#     angle = math.atan(slope)
#     print("angle", angle)
#
#     a = math.sin(angle) * (distance/30)
#     print("a", a)
#     b = math.cos(angle) * (distance/30)
#     print("b", b)
#
#     x_a = point_one[0] + a
#     y_b = point_one[1] + b
#
#     print("New points", (int(x_a), int(y_b)))
#     new_distance = math.sqrt((int(x_a) - point_two[0]) ** 2 + (int(y_b) - point_two[1]) ** 2)
#     print("new distance", new_distance)
#     return int(x_a), int(y_b)

def new_coordinates(point_one, point_two):
    """
    Based on "The intercept theorem", also known as "Thales' theorem"
    https://en.wikipedia.org/wiki/Intercept_theorem

    """

    dx = (point_two[0] - point_one[0])
    dy = (point_two[1] - point_one[1])

    x_a = point_one[0] + dx/20
    y_b = point_one[1] + dy/20

    print("New points", (int(x_a), int(y_b)))

    return int(x_a), int(y_b)

def find_line_fit(slope_intercept):
    """slope_intercept is an array [[slope, intercept], [slope, intercept]...]."""

    # Initialise arrays
    kept_slopes = []
    kept_intercepts = []
    # print("Slope & intercept: ", slope_intercept)
    if len(slope_intercept) == 1:
        return slope_intercept[0][0], slope_intercept[0][1]

    # Remove points with slope not within 1.5 standard deviations of the mean
    slopes = [pair[0] for pair in slope_intercept]
    mean_slope = np.mean(slopes)
    slope_std = np.std(slopes)
    for pair in slope_intercept:
        slope = pair[0]
        # print(slope - mean_slope, 1.5 * slope_std)
        if slope - mean_slope < 1.5 * slope_std:
            kept_slopes.append(slope)
            kept_intercepts.append(pair[1])
    if not kept_slopes:
        kept_slopes = slopes
        kept_intercepts = [pair[1] for pair in slope_intercept]
    # Take estimate of slope, intercept to be the mean of remaining values
    slope = np.mean(kept_slopes)
    intercept = np.mean(kept_intercepts)
    # print("Slope: ", slope, "Intercept: ", intercept)
    return slope, intercept

# Getting the image
try:
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)
# plt.imshow(image)

gray_image = grayscale(image)

kernel_size = 5
gaussian_blur_image = gaussian_blur(gray_image, kernel_size)

low_threshold = 50
high_threshold = 150
edges_image = canny(gaussian_blur_image, low_threshold, high_threshold)

# Masking the image
imshape = image.shape
vertices = np.array([[(50, imshape[0]), (400, 340), (560, 340), (imshape[1], imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges_image, vertices)

# Applying Hough transform to masked image
rho = 1
theta = np.pi/180
threshold = 10
min_line_length = 10
max_line_gap = 2
hough_lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len=min_line_length, max_line_gap=max_line_gap)

combo_image = weighted_img(hough_lines_image, image)
# Display images
# images = [hough_lines_image, masked_edges, combo_image]
# for ima in images:
#     plt.figure()
#     plt.imshow(ima)

f = plt.figure()

f.add_subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original image')

f.add_subplot(2, 2, 2)
plt.imshow(masked_edges, cmap='gray')
plt.title('Masked image')

f.add_subplot(2, 2, 3)
plt.imshow(hough_lines_image, cmap='Greys_r')
plt.title("Canny edges of Gaussian image")

f.add_subplot(2, 2, 4)
plt.imshow(combo_image)
plt.title("Hough transformed image of Canny edges")
plt.show()
