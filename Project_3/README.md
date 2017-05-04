# **Behavioral Cloning**

**Table of Content**

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [1 Preparing the data](#1-preparing-the-data)
	- [1.1 Details of `preprocessor_modeler.py`](#11-details-of-preprocessormodelerpy)
- [2 Data Modelling](#2-data-modelling)
	- [2.1 Details of `model.py`](#21-details-of-modelpy)
- [3 Loss Plots](#3-loss-plots)
	- [3.1 Details of `plots.py`](#31-details-of-plotspy)
- [4 Autonomous Driving](#4-autonomous-driving)
- [5 Discussion](#5-discussion)

<!-- /TOC -->

In this project, I have tried to clone the behavior of the car following a path, he has used [Udacity's Self-Driving Car Simulator v2](https://github.com/udacity/self-driving-car-sim) to record the path while its been used.

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

In this project, I would consider `[center, left and right]` as `x_train` and the `left` angles as `y_train`.

![Merged first frame](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/merge_frame.jpg)
First frame of `left`, `center` and `right` after the data has been acquired, whos data is given by:

| Left File Name                   | Center File Name                   | Right File Name                   | Left Angle | Acceleration | Right Angle | Speed |
|----------------------------------|------------------------------------|-----------------------------------|------------|--------------|-------------|-------|
| left_2017_04_28_16_17_58_298.jpg | center_2017_04_28_16_17_58_298.jpg | right_2017_04_28_16_17_58_298.jpg | 0          | 0            | 0           | 0     |

---

I have split this project into four sections:

1. Preparing the data
2. Data modelling
3. Plotting the loss
4. Data prediction (regression)

**File Structure**

* `drive.py` - A server to send regression angles to the simulator.
* `preprocessor_modeler.py` - Read the data from the folder, preprocess them and yeal as generators for batch processing.
* `model.py` - Create model for each epoch, save the loss values and the final model.
* `plots.py` - To plot the loss values.
* `model.json` - Summary of the model in JSON format.
* `README.md` - Detailed explanation of the project.

## 1 Preparing the data

As mentioned earlier, I used [Udacity's Self-Driving Car Simulator v2](https://github.com/udacity/self-driving-car-sim) to collect the data, he took six laps across the simulator and tried his best to keep the car in the center of the road. One lap would not be enough for the network to be trained and it is not consistent enough.

The data was not converted into grayscale because I think that the color aspect is an import feature for any neural network to work.

![Random shear](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/random_shear.png)

Above image shows a random image that is sheared and cropped.

### 1.1 Details of `preprocessor_modeler.py`

Before the data is trained, I maid sure that the data was preprocessed. This preprocessing of the data had the following flow:

1. Based on `Bernoulli trail`, if I get `1` a shear is added to the image.
2. Then, crop 35% from top and 10% from bottom.
3. Again based on `Bernoulli trail`, flip the image and its steering angle.
4. Then add random brightness using lookup table.
5. Finally, resize the image to `(64, 64)`, to make the data smaller.

## 2 Data Modelling

`model.py` contains the code for modelling that uses CNN, whose summery is given by:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 16)        3088
_________________________________________________________________
elu_1 (ELU)                  (None, 16, 16, 16)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 32)          12832
_________________________________________________________________
elu_2 (ELU)                  (None, 8, 8, 32)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 64)          51264
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0
_________________________________________________________________
elu_3 (ELU)                  (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800
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
Total params: 617,685
Trainable params: 617,685
Non-trainable params: 0
_________________________________________________________________
```

I have used slightly changed architecture comma.ai and with the help of Keras `fit_generator` method, I was able to do batch training.

### 2.1 Details of `model.py`

Following is the description of `model.py` file:

The model is initalised using Keras `Sequential` layer, then adding to it, the data is normalised and the input shape is given. Three convolution layers are added, two of which has `ELU` activation layer then the next layer goes through 2D convolution and finally flattened with `ELU` activation layer & `Dropout` over fitting function.

Then, three fully connected layers are added, of which the first layer has `512` neuron, then `50` neurons and finally `1` output layer.

The Neural Network (NN) is compiled with `Adam` optimiser, Mean Squared Error (MSE) loss calculating function and finally, instructed to calculate the accuracy of the result.

Models summary is printed out using `model.summary()`, where `model` is an instantiated variable of `Sequential`.

Data for training and validation is initialised, that are python `generators`, befor this an custom object `LossHistory` object is initialised, which logs all the loss that is generated by the `fit` method.

The model is trained using Keras `fit_generator` method, with `8` epochs; each epoch has `20032` steps and `6400` validation steps. The custom loss logging method is given to this `fit` method.

Once the model is trained and validated, the model is exported as `model.h5`, its network configurations are saved as `model.json` and the loss history is saved as `model_loss.csv`.

**Summary of Model**

| Layer                   | Details                                                                                  |
|-------------------------|------------------------------------------------------------------------------------------|
| Convolution Layer 1     | Filter: 16 <br> Kernal: 8-by-8 <br> Stride: 4-by-4<br> Padding: SAME<br> Activation: ELU |
| Convolution Layer 2     | Filter: 32<br> Kernal: 5-by-5<br> Stride: 2-by-2<br> Padding: SAME<br> Activation: ELU   |
| Convolution Layer 3     | Filter: 64<br> Kernal: 5-by-5<br> Stride: 2-by-2<br> Padding: SAME<br> Activation: ELU   |
| Flatten layer           | Dropout: 0.2<br> Activation: ELU                                                         |
| Fully Connected Layer 1 | Neurons: 512<br> Dropout: 0.5<br> Activation: ELU                                        |
| Fully Connected Layer 1 | Neurons: 50<br> Activation: ELU                                                          |
| Fully Connected Layer   | Neurons: 1<br> Activation: linear                                                          |

## 3 Loss Plots
Loss for `8` epochs is plotted as:

![100 epochs loss](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/epochs.png)

and the average of `8` epochs is given by:
![Average of 8 epochs](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/average_loss.png)

You can see that, from the above plot, there is a sudden dip at epoch `1` and gradually drops down, which shows that the model is able to train well.

Also, the normal distribution of the steering angel is given by:

![Histogram of steering angel](https://github.com/akshaybabloo/Car-ND/raw/master/Project_3/assets/norm.png)

### 3.1 Details of `plots.py`

`plots.py` is used to plot the loss of the model and the distribution of the data.

`model_loss.csv` is read as Pandas `DataFrame` and is plotted as is. Next, the same data is averaged with the number of steps, i.e. `20032`, and plotted.

Also, I tried to plot the random normal distribution of the steering angels.

## 4 Autonomous Driving

Using slightly changed version of Udacity's `drive.py`, the `model.h5` is given to it so that the `predict` method of Keras can predict the images from the saved model.

The only thing that was changed is that, the `images` are matched with the models data, i.e. the model is cropped and resized before giving it to the `predict` method.

## 5 Discussion

* I have trained the model with `6` laps of the simulation, the model could have been much better if the data was modeled on a better architecture.
* In future I would like to use `ResNet` or `NVIDIA's` architecture.
