# Advanced Line Detection

**Table of Content**

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [1 Calibrate the Camera](#1-calibrate-the-camera)
- [2 Change the Perspective](#2-change-the-perspective)
- [3 Generate Lane Masking](#3-generate-lane-masking)
- [4 Detecting Lanes](#4-detecting-lanes)

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

Using OpenCV, the detection of the lanes are done as follows:



## 4 Detecting Lanes
