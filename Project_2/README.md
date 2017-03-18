# Traffic Sign Recognition

**Table of Content**

- [Data Set Summary & Exploration](#data-set-summary-exploration)
	- [Data Structure](#data-structure)
	- [Dataset sample](#dataset-sample)
- [Design and Test a Model Architecture](#design-and-test-a-model-architecture)
	- [Data Preprocessing](#data-preprocessing)
	- [Training](#training)
	- [Final Architecture](#final-architecture)
	- [Details for training the model](#details-for-training-the-model)
	- [Results](#results)
- [Test a Model on New Images](#test-a-model-on-new-images)
	- [Images used](#images-used)
	- [Prediction](#prediction)
	- [Softmax probabilities](#softmax-probabilities)
- [Reference](#reference)

This project involves in detecting the German traffic signs using the principals of Convolutional Neural network (CNN).

Before I go in the details of how the algorithm works, lets look at the images used to classify the trained model.

---

The project code can be found at [https://github.com/akshaybabloo/Car-ND/tree/master/Project_2/](https://github.com/akshaybabloo/Car-ND/tree/master/Project_2/)

### Data Set Summary & Exploration

The dataset used in this project is a public dataset by [Institute Fur Neuroinformatik Benchmark](http://benchmark.ini.rub.de/), that has been downsized to fit the project need.

#### Data Structure

The code for this step is contained in the cell `60` of the Jupyter notebook.  

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* Number of training samples = 34799
* Number of testing samples = 12630
* Number of validation samples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

The number of unique samples is given by the following bar graph:

![Unique samples](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/unique.JPG)

Where, the x axis represents the class labels and the y axis represents the number of samples associated to a particular class.

#### Dataset sample

The sample of the dataset can be found in the cell `63, 64` and `65` representing train, test and validation respectively.

![Sample](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/sample.png)

### Design and Test a Model Architecture

#### Data Preprocessing

In cell `66` the data was normalised but not grey scaled, that's because I feel that the colour makes an important feature in the case of signs because what if the signs are inverted and somehow colour could be an import way to recognise the sign.

Sample of the normalised image:

![Normalised Image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/normalised.png)

#### Training

The dataset has already been split into training, validation and testing. The only thing I had to do was to shuffle them (cell `76`) randomly by using `scikit-learn's` `shuffle`.

The training model is based up on `LeNet` architecture, which looks something like

![LeNet](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/lenet.png)

The code for `LeNet` can be found in cell `77`, which uses two convolutions and three fully connected network.



Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


#### Final Architecture

My final model consisted of the following layers:

| Layers            | Inputs and Outputs                   | Computation                                                                                                                                                                                                                                                                                                  |
|-------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Convolution 1     | Inputs: 32x32x1<br> Output: 28x28x6  | **Truncated Normal** <br> Mean: 0<br> Standard Devation: 0.1<br> Shape: (5, 5, 3, 6)<br><br> **2D Convolutions**<br> strides: [1, 1, 1, 1]<br> padding: VALID<br><br> **ReLU Activation**<br><br> **Max Pooling**<br> padding: VALID<br> ksize: [1, 2, 2, 1]<br> strides: [1, 2, 2, 1]<br> padding: VALID    |
| Convolution 2     | Inputs: 14x14x6<br> Output: 10x10x16 | **Truncated Normal**<br><br> Mean: 0<br> Standard Devation: 0.1<br> Shape: (5, 5, 3, 6)<br><br> **2D Convolutions**<br> strides: [1, 1, 1, 1]<br> padding: VALID<br><br> **ReLU Activation**<br><br> **Max Pooling**<br> padding: VALID<br> ksize: [1, 2, 2, 1]<br> strides: [1, 2, 2, 1]<br> padding: VALID |
| Flatten (Reshape) | Input: 10x10x16<br> Output: 400      |                                                                                                                                                                                                                                                                                                              |
| Fully Connected 1 | Input: 400<br> Output: 120           | **Truncated Normal**<br> shape: (400, 120)<br> mean: 0<br> sigma: 0.1<br><br>  **WX+B**<br><br>  **ReLU Activation**                                                                                                                                                                                         |
| Fully Connected 2 | Input: 120<br> Output: 84            | **Truncated Normal**<br> shape: (120, 84)<br> mean: 0<br> sigma: 0.1<br><br> **WX+B**<br><br> **ReLU Activation**                                                                                                                                                                                            |
| Output            | Input: 84<br> Output:43              | **Truncated Normal**<br> shape: (84, 43)<br> mean: 0<br> sigma: 0.1<br><br> **WX+B**<br><br> **ReLU Activation**                                                                                                                                                                                             |

#### Details for training the model

The hyperparameters used for training the model are as follows:

* Batch size: 100
* Number of epochs: 100
* Learning rate: 0.001
* Mean for Truncated Normal: 0
* Standard deviation for Truncated Normal: 0.1
* Convolution type: VALID

I have trained the model using the batch size ranging between 1 to 200. If the batch size is too small the time take for training the model seems to be high, though the time taken to train decreases significantly if the batch size is large, but the model is instable i.e. the validation accuracy is significantly less.

I was able to get an optimum validation accuracy when the batch size was set to `100`.

The validation graph for `1` to `200` is:

![Validation graph](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/validation_graph.png)

#### Results

The code for calculating the accuracy of the model is located in the cell `80` of the Jupyter notebook.

My final model results were:
* Test Accuracy = 0.929
* Validation Accuracy = 0.950

If a well known architecture was chosen:

As mentioned earlier LeNet [1] architecture was chosen as it is a well known CNN for MNIST dataset.

This approach got me an accuracy result of `95%`, which is not that good but does a good job.

### Test a Model on New Images

#### Images used

![10 KM](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/10.jpg)

![Give way](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/give_way_sign.jpg)

![No entry](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/no_entry.jpg)

![Stop](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/stop.jpg)

![Wrong way](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/wrong_way.jpg)

These images are not German traffic signs, but they are signs from USA, UK and NZ.

#### Prediction

The code for making predictions on my final model is located in the cell `93` of the Jupyter notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No Entry      		| No Entry   									|
| Speed limit (10km/h) | Speed limit (20km/h) |
| Wrong Way					| Stop											|
| Give Way	      		| Yield					 				|
| Stop			| Stop      							|


The model was able to correctly guess 2 of the 5 traffic signs.

#### Softmax probabilities

The code for making predictions on my final model is located in the cell `130` of the Jupyter notebook.

For the first image, that detected the as `No Entry`

**Image 1**

![Image 1 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image1_p.png)

* 13 :  Yield
* 36 :  Go straight or right
* 20 :  Dangerous curve to the right
* 5 :  Speed limit (80km/h)
* 30 :  Beware of ice/snow

**Image 2**

![Image 2 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image2_p.png)

* 16 :  Vehicles over 3.5 metric tons prohibited
* 27 :  Pedestrians
* 34 :  Turn left ahead
* 6 :  End of speed limit (80km/h)
* 19 :  Dangerous curve to the left

**Image 3**

![Image 3 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image3_p.png)

* 1 :  Speed limit (30km/h)
* 30 :  Beware of ice/snow
* 16 :  Vehicles over 3.5 metric tons prohibited
* 41 :  End of no passing
* 24 :  Road narrows on the right

**Image 4**

![Image 4 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image4_p.png)

* 30 :  Beware of ice/snow
* 27 :  Pedestrians
* 4 :  Speed limit (70km/h)
* 19 :  Dangerous curve to the left
* 8 :  Speed limit (120km/h)

**Image 5**

![Image 5 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image5_p.png)

* 23 :  Slippery road
* 31 :  Wild animals crossing
* 42 :  End of no passing by vehicles over 3.5 metric tons
* 6 :  End of speed limit (80km/h)
* 17 :  No entry

## Reference

[1] LeCun, Y., Jackel, L. D., Bottou, L., Brunot, A., Cortes, C., Denker, J. S., ... & Simard, P. (1995, October). Comparison of learning algorithms for handwritten digit recognition. In International conference on artificial neural networks (Vol. 60, pp. 53-60).
