# **Behavioral Cloning**

**Table of Content**

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [1 Preparing the data](#1-preparing-the-data)
- [2 Data Modelling](#2-data-modelling)
	- [2.1 Details of `model.py`](#21-details-of-modelpy)
	- [2.2 Loss Plots](#22-loss-plots)

<!-- /TOC -->

In this project the author has tried to clone the behavior of the car following a path, he has used [Udacity's Self-Driving Car Simulator v2](https://github.com/udacity/self-driving-car-sim) to record the path while its been used.

The simulator, once the recording is completed, outputs the camera frames as following files:

```
+-- IMG
|    |
|    +-- center_*.jpg
|    +-- ..
|    +-- right_*.jpg
|    +-- ..
|    +-- left_*.jpg
|    `-- ..
`-- driving_log.csv
```

`center_*.jpg`, `right_*.jpg` and `left_*.jpg` are the camera angles (3 cameras placed on left, right and center of the car.). `driving_log.csv` has seven columns, describing as follows:

1. Column 1 - Location and name of `center` frame
2. Column 2 - Location and name of `left` frame
3. Column 3 - Location and name of `right` frame
4. Column 4 - Angle to turn `left`
5. Column 5 - Acceleration
6. Column 6 - Angle to turn `right`
7. Column 7 - Speed

In this project the author would consider `[center, left and right]` as `x_train` and the `left` angles as `y_train`.

![Merged first frame](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/merge_frame.jpg)
First frame of `left`, `center` and `right` after the data has been acquired, whos data is given by:

| Left File Name                   | Center File Name                   | Right File Name                   | Left Angle | Acceleration | Right Angle | Speed |
|----------------------------------|------------------------------------|-----------------------------------|------------|--------------|-------------|-------|
| left_2017_04_28_16_17_58_298.jpg | center_2017_04_28_16_17_58_298.jpg | right_2017_04_28_16_17_58_298.jpg | 0          | 0            | 0           | 0     |

---

The author has split this project into four sections:

1. Preparing the data
2. Data modelling
3. Plotting the loss
4. Data prediction (regression)

**File Structure**

* `drive.py` - A server to send regression angles to the simulator.
* `make_pickle.py` - Read the data from the folder and pickle them into `data.p`.
* `model.py` - Create model for each epoch, save the loss values and the final model.
* `plot_loss.py` - To plot the loss values.
* `model.json` - Summary of the model in JSON format.
* `README.md` - Detailed explanation of the project.

## 1 Preparing the data

As mentioned earlier, the author used [Udacity's Self-Driving Car Simulator v2](https://github.com/udacity/self-driving-car-sim) to collect the data, he took six laps across the simulator and tried his best to keep the car in the center of the road. One lap would not be enough for the network to be trained and it is not consistent enough.

The data was not converted into grayscale because the author thinks that the color aspect is an import feature for any neural network to work.

The next step was to augment the data to remove any noise, for this to happen the frames were flipped vertically and appended to the original data, which doubled the dataset to `13658` samples.

![Flipped image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/merge_flipped.jpg)
First frame of `left`, `center` and `right` flipped.

These samples were then pickled into one file and named it as `data.p`.

## 2 Data Modelling

`model.py` contains the code for modelling that uses CNN, whose summery is given by:


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 17, 80, 16)        3088
_________________________________________________________________
elu_1 (ELU)                  (None, 17, 80, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 40, 32)         12832
_________________________________________________________________
elu_2 (ELU)                  (None, 9, 40, 32)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 20, 64)         51264
_________________________________________________________________
flatten_1 (Flatten)          (None, 6400)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 6400)              0
_________________________________________________________________
elu_3 (ELU)                  (None, 6400)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               3277312
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
elu_4 (ELU)                  (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                25650
_________________________________________________________________
elu_5 (ELU)                  (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 51
=================================================================
Total params: 3,370,197
Trainable params: 3,370,197
Non-trainable params: 0
_________________________________________________________________
```

### 2.1 Details of `model.py`

Following is the description of `model.py` file:

* From line `15` to `31`, its a custom class to log all the loss vales while training the data.
* From line `35` to `36`, reading the pickled data.
* AT line `38`, the author is loading `x_train` and `y_train`
* At line `41`, Keras `Sequential` model is initialised.
* At line `44`, Keras `Lambda` class is added to write a Python's `lambda` function to normalise the data.
* At line `46`, unwanted content of the image is cropped, `70` pixels of upper and `25` pixels of the bottom image is removed.
* From line `49` to `57`, Keras convolutions are used that has:

| Layer                   | Details                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------|
| Convolution Layer 1     | Filter: 16 <br> Kernal: 8-by-8 <br> Stride: 4-by-4<br> Padding: SAME<br> Activation: ELU |
| Convolution Layer 2     | Filter: 32<br> Kernal: 5-by-5<br> Stride: 2-by-2<br> Padding: SAME<br> Activation: ELU   |
| Convolution Layer 3     | Filter: 64<br> Kernal: 5-by-5<br> Stride: 2-by-2<br> Padding: SAME<br> Activation: ELU   |
| Flatten layer           | Dropout: 0.2<br> Activation: ELU                                                         |
| Fully Connected Layer 1 | Neurons: 512<br> Dropout: 0.5<br> Activation: ELU                                        |
| Fully Connected Layer 1 | Neurons: 50<br> Activation: ELU                                                          |
| Fully Connected Layer   | Neurons: 1<br> Activation: linear                                                          |

* At line `75`, Keras `Adam` optimiser is used with learning rate of `0.0001`.
* At line `77`, the learning is configured using `Mean Squared Error` for calculating the loss and to calculate the accuracy.
* From line `82` to `85`, at every epoch, a model is saved.
* Line `87`, initialises custom history to save loss.
* At line `88`, the model is trained for `100` epochs, with a validation split of `0.2`, shuffling is enabled.
* from `90` to `105`, the loss is saved to `model_loss.csv`, total model configuration is saved as JSON file (`model.json`), final model weights are saved to `model_weights.h5` and finally the complete model is saved to `model.h5`

### 2.2 Loss Plots
Loss for `100` epochs is plotted as:
![100 epochs loss](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/100_epochs.png)

and the average of `100` epochs is given by:
![Average of 100 epochs](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/average_loss.png)

You can see that, from the above plot, the data converges at epochs `77` to `100`, the average loss study between `0.0013` to `0.0012`
