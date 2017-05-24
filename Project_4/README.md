# Advanced Line Detection

**Table of Content**

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [1 Calibrate the Camera](#1-calibrate-the-camera)
- [2 Change the Perspective](#2-change-the-perspective)
- [3 Generate Lane Masking](#3-generate-lane-masking)
	- [3.1 Step-by-step Explanation](#31-step-by-step-explanation)
	- [3.2 Add Perspective to the Image](#32-add-perspective-to-the-image)
- [4 Detecting Lanes](#4-detecting-lanes)
- [5 Discussion](#5-discussion)

<!-- /TOC -->

In this project, I have tried to detect the road lanes and draw a polygon on all the detected lanes. The program is divided into the following sections

1. Calibrate the Camera
2. Change the Perspective
3. Generate Lane Masking
4. Detecting Lanes

## 1 Calibrate the Camera

Using the chessboard images provided by Udacity ([https://github.com/udacity/CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines)), I have implemented a way to calibrate the input image and undistort them.

To calibrate the camera, I have used `OpenCV` internal function to detect the chessboard points (corners). If the chessboard corners are found, the lines are drawn from point-to-point.

![Chessboard](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/chessboard.png)

![Undistorted chessboard](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/undistort_chess.png)

This calibration can be applied to an image to undistort the image, which would look like:

![Undistorted image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/undistort.png)

## 2 Change the Perspective

Changing the perspective of an image can make simple adjustments easy; for example lets consider image

![Undistorted image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/test2.jpg)

Changing its perspective to something like

![perspective image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/perspective.png)

This perspective helps in detecting lines (easily) as if we are looking at it from above (birds eye view).

## 3 Generate Lane Masking

Let's look at the image step-by-step without perspective transformation:

![Extracted lines](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/extract_lines_no_per.png)

### 3.1 Step-by-step Explanation

1. First, read the image and remove the first `400` vertical pixels.
2. After removing the image's pixels, Convert the image to `YUV` color space, where `Y` represents the brightness, `U` represents difference of `blue` & `brightness` (B-Y) and `V` represents the difference of `red` & `brightness` (R-Y)
3. Then remove the `RED` hue from the `YUV` color space.
4. Now, convert the original image to `HLS` color space, where `H` represents `Hue`, `L` represents `Lightness` and `S` represents `Saturation`.
5. Stack the `brightness` & difference of `blue-brightness` of `YUV` with `HLS`'s `Lightness`.
6. Convert the stacked image to grayscale.
7. Take the Sobel derivative of the gray scaled image in `X` direction.
8. Take the Sobel derivative of the gray scaled image in `Y` direction.
9. Calculate the direction of the gradient for `X` and `Y` Sobel.
10. Calculate the gradient magnitude of Sobel `X` and `Y`.
11. Extract `Yellow` pixels from the original image.
12. Extract `Red` highlights from original image.
13. Create a binary image based on thresholds.
14. Reduce binary noise.

### 3.2 Add Perspective to the Image

As seen in [2 Change the Perspective](#2-change-the-perspective), let's add some perspective to the image so that it could be manipulated later. The following is it's result:

![Perspective of lines](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/perspective_extracted.png)

## 4 Detecting Lanes

| Left Line                                                                                                 | Detected Lane                                                                                              | Right Lane                                                                                                  |
|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| ![Left lines hist](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/left_line_hist.png) | ![Extracted lines](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/extracted_lines.png) | ![Right lines hist](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/right_line_hist.png) |

From the above image, we can see that the histogram on left and right side represents the position at which the bright pixels were found. This function returns the pixel value at which the bright pixels are found, keeping these positions as is a line is drawn between them.

![Perspective of Lane drawn](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/line_poly.png)

The above image shows a poly drawn over the perspective image. You can also see that the top right line seems to be out of alignment, this happens because the detection of line is by chance and number of iterations for the histogram search for the lines. Higher the iterations the better the line fits.

Converting the perspective to the normal and adding some information about the image, we would get something like this:

![Car ND](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/car-nd.png)

## 5 Discussion

* If you look at cell `21`of the Jupyter notebook, you can see that the `fifth` row, first from left, could not find the binary threshold of the left, the `brightness` was able to detect the a dot, because it didn't have any previous information of the frame before the current frame, it couldn't add a straight line.
* If you see the `seventh` figure of cell `21`, you can see that there are two yellow lines, maybe this problem could be solved by filling the dark spots between the two lines with yellow (or any color that is recognised)
