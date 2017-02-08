# Self-Driving Car Engineer Nanodegree

Udacity "Self-Driving Car Engineer" Nanodegree files

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [File Structure](#file-structure)
- [Documentations](#documentations)
- [Term 1](#term-1)
	- [Finding Lanes](#finding-lanes)
- [License and Disclaimer](#license-and-disclaimer)

<!-- /TOC -->

## File Structure

```
Car-ND
|   .gitignore
|   LICENSE
|   README.md
|
\---Term-1
    \---Finding_Lane_Lines
        |   1_get_bright_lanes.py
        |   2_color_region_masking.py
        |   3_canny_detection.py
        |   4_hough_transform.py
        |   5_hough_transform_masked.py
        |   test.jpg
        |
        \---P1
            |   challenge.jpg
            |   challenge.mp4
            |   extra.mp4
            |   Project-1.ipynb
            |   requirements.txt
            |   solidWhiteRight.mp4
            |   solidYellowLeft.mp4
            |   white.mp4
            |   white_no_slope.mp4
            |   yellow.mp4
            |
            \---test_images
                    solidWhiteCurve.jpg
                    solidWhiteRight.jpg
                    solidYellowCurve.jpg
                    solidYellowCurve2.jpg
                    solidYellowLeft.jpg
                    whiteCarLaneSwitch.jpg
```

## Documentations

To build Python documentations

```
make html
```

To rebuild the documentations

```
make clean
make html
```

## Term 1

Term 1 is mainly focused on finding the lanes and objects correctly using various algorithms and computer vision.

### Finding Lanes

In this section we were introduced to the basic knowledge of Computer Vison using [OpenCV](http://opencv.org/) and [Python](https://www.python.org/download/releases/3.0/). The files are available [here](https://github.com/akshaybabloo/Car-ND/tree/master/Term-1/Finding_Lane_Lines).

More on this in my blog [Part 1 Car-ND: Detect road lanes using Computer Vision and Python 3](https://blog.gollahalli.com/blog/29/1/2017/part-1-1-car-nd-detect-road-lanes-using-computer-vision-and-python)

**Outcome**

![Road lanes](https://github.com/akshaybabloo/Car-ND/raw/master/Screenshots/road_lanes.png)

## License and Disclaimer

All product and company names are trademarks™ or registered® trademarks of their respective holders. Use of them does not imply any affiliation with or endorsement by them. Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/), and code samples are licensed under the [MIT License](https://github.com/akshaybabloo/Car-ND/blob/master/LICENSE).
