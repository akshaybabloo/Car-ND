# Advanced Line Detection

**Table of Content**

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [1 Calibrate the Camera](#1-calibrate-the-camera)
- [2 Change the Perspective](#2-change-the-perspective)
- [3 Generate Lane Masking](#3-generate-lane-masking)
- [4 Draw the Polys](#4-draw-the-polys)

<!-- /TOC -->

In this project, I have tried to detect the road lanes and draw a polygon on all the detected lanes. The program is divided into the following sections

1. Calibrate the Camera
2. Change the Perspective
3. Generate Lane Masking
4. Draw the Polys

## 1 Calibrate the Camera

Using the chessboard images provided by Udacity ([https://github.com/udacity/CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines)), I have implemented a way to calibrate the input image and undistort them.

To calibrate the camera, I have used `OpenCV` internal function to detect the chessboard points (corners). If the chessboard corners are found, the lines are drawn from point-to-point.

![Chessboard](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/chessboard.png)

This calibration can be applied to an image to undistort the image, which would look like:

![Undistorted image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_4/assets/undistort.png)

## 2 Change the Perspective


## 3 Generate Lane Masking


## 4 Draw the Polys
