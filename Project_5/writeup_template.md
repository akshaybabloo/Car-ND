# Vehicle Detection and Tracking

**Table of Content*

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Content of this repo](#content-of-this-repo)
- [Classifier](#classifier)
- [Sliding Window and the classifier testing](#sliding-window-and-the-classifier-testing)
	- [Implemented ideas:](#implemented-ideas)
- [The pipeline visualisation](#the-pipeline-visualisation)
- [Results and discussion](#results-and-discussion)

<!-- /TOC -->

## Content of this repo

- `VehicheDetect.ipynb` - Jupyter notebook with code for the project
- `test_images` - a directory with test images
- `project_video_proc.mp4` - the result video
- `project_video.mp4` - the original raw video

## Classifier

Features are needed to train a classifier and make predictions on the test or real-world images.

The project required to build a classifier that is able to answer if there is a car in a given image ( a subset of the whole image). To address this task three types of features were used: HOG (Histogram of Oriented Gradients) (shape features), binned colour (colour and shape features) and colour histogram features (colour only features). This combination of features can provide enough information for image classification.

Firstly, an automated approach was applied to tune the HOG parameters (`orientations, pixels_per_cell, cells_per_block`).

Something like:
```Python
from skopt import gp_minimize
space  = [(8, 64),                  # nbins
          (6, 12),                  # orient
          (4, 16),                   # pix_per_cell
          (1, 2)]                   # cell_per_block
i = 0
def obj(params):
    global i
    nbins, orient, pix_per_cell, cell_per_block = params
    car_features = extract_features(cars[0:len(cars):10], nbins, orient, pix_per_cell, cell_per_block)
    notcar_features = extract_features(notcars[0:len(notcars):10], nbins, orient, pix_per_cell, cell_per_block)
    y = np.hstack((np.ones(len(cars[0:len(cars):10])), np.zeros(len(notcars[0:len(notcars):10]))))
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    test_acc = svc.score(X_test, y_test)
    print i, params, test_acc
    i+=1
    return 1.0-test_acc

res = gp_minimize(obj, space, n_calls=20, random_state=22)
"Best score=%.4f" % res.fun
```

However, results were not very good because it ended with high numbers for HOG parameters which result in very slow feature extraction with comparable to less computational expensive parameters set accuracy. That is why the parameters for HOG as well as parameters for other features extractors were fine tuned manually by trial and error process so that it optimises accuracy and computation time.

Here is an example of a train image and its HOG:

![Example image](assets/ex.jpg) ![HOG of example image](assets/hog.jpg)

Final parameter for feature extraction:

```Python
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

Normalising ensures that a classifier's behaviour isn't dominated by just a subset of the features and that the training process is as efficient as possible. That is why feature list was normalized by the `StandardScaler()` method from `sklearn`. The data is split into training and testing subsets (80% and 20%). The classifier is a linear SVM. It was found that it performs well enough and quite fast for the task. The code chunk under *Classifier* represents these operations.


## Sliding Window and the classifier testing

Basic sliding window algorithm was implemented in the same way to one presented in Udacity's lectures (See the code chunks under *Slide window* header). It allows searching a car in the desired region of the frame with the desired window size (each subsampled window is rescaled to 64x64 px before classified by the SVC).

The window size and overlap should be wisely selected. The size of the window should be compared to the size of an expected car. These parameters were set to mimic perspective.

There are some sample results for a fixed window size (128x128 px) and overlap for the provided test images:

![Test image 1](output_images/test1.jpg)
![Test image 2](output_images/test2.jpg)
![Test image 3](output_images/test3.jpg)
![Test image 4](output_images/test4.jpg)
![Test image 5](output_images/test5.jpg)
![Test image 6](output_images/test6.jpg)

As we can see on examples above, the classifier successfully finds cars on the test images. However, there is a false positive example, so, we will need to apply a kind of filter (such as heat map) and the classifier failed to find a car on the 3rd image because it is too small for it. That is why we will need to use multiscale windows.

### Implemented ideas:

- To increase the classifier accuracy, feature extraction parameters were tuned. The data was augmented by flipped images.

- To reduce the number of false positives a heat map with a threshold approach was implemented in the same to the suggested in the lectures way. For video, the heat map is accumulated by two frames which reduce the number of outliers false positives.

- To increase performance it is needed to analyze the smallest possible number of windows. That is why, one can scan with a search window, not across the whole image, but only areas where a new car can appear and also we are going to scan areas where a car was detected (track cars)

*There is an example of a new car detection ROI:*

![New car detection ROI](assets/new_car_windows.jpg)

*And a car tracking ROI:*

![New car detection ROI](assets/car_track_window.jpg)

- It is important to use a different scale of the classifiers window on different parts of the image due to perspective. So, different ROI  window sizes were applied on different areas (realised in the `frame_proc` function).

- In order to reduce jitter the function `filt` applies a simple low-pass filter on the new and the previous cars boxes coordinates and sizes (see under the *Frames processing* header) with weight `ALPHA=0.75` of the previous data. This makes car boundaries on video quite smooth.

- To increase performance the analysis was skipped for every 2nd frame because we do not expect very fast moving of the detected cars. Known cars boundaries from the previous frame are used in such cases.

## The pipeline visualisation

Areas of interest for tracking of detected cars are marked green. Hot windows (which were classified as cars) are yellow.

![Image proc](assets/image_proc.jpg)

The heat map of found hot windows overlap:

![Heatmap](assets/heatmap.jpg)

The final result of an image with cars boundaries.

![Image proc](assets/image_fin.png)


## Results and discussion

The pipeline is able to correctly label cars areas on video frames.

- Of course, the algorithm may fail in case of difficult light conditions, which could be partly resolved by the classifier improvement.

- It is possible to improve the classifier by additional data augmentation, hard negative mining, classifier parameters tuning etc.

- The algorithm may have some problems in case of car overlaps another. To resolve this problem one may introduce long-term memory of car position and a kind of predictive algorithm which can predict where an occluded car can be and where it is worth to look for it.

- To eliminate false positives on areas out of the road, one can deeply combine results from the Advanced Lane Line finding project to correctly determine the wide ROI on the whole frame by the road boundaries. Unfortunately, it was not correctly implemented (just hard coded, which is enough for the project but not a good implementation for a real-world application) due to time limitation.

- The pipeline is not a real-time. One can further optimise a number of features and feature extraction parameters as well as a number of analysed windows to increase the rate.
