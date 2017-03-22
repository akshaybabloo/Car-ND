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

The code for this step is contained in the cell `61` of the Jupyter notebook.  

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

The sample of the dataset can be found in the cell `64, 65` and `66` representing train, test and validation respectively.

![Sample](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/sample.png)

### Design and Test a Model Architecture

#### Data Preprocessing

In cell `67` the data was normalised but not grey scaled, that's because I feel that the colour makes an important feature in the case of signs because what if the signs are inverted and somehow colour could be an import way to recognise the sign.

Sample of the normalised image:

![Normalised Image](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/normalised.png)

#### Training

The dataset has already been split into training, validation and testing. The only thing I had to do was to shuffle them (cell `70`) randomly by using `scikit-learn's` `shuffle`.

The training model is based up on `LeNet` architecture, which looks something like

![LeNet](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/lenet.png)

The code for `LeNet` can be found in cell `71`, which uses two convolutions and three fully connected network.



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

The code for calculating the accuracy of the model is located in the cell `75` of the Jupyter notebook.

My final model results were:
* Test Accuracy = 0.935
* Validation Accuracy = 0.951

If a well known architecture was chosen:

As mentioned earlier LeNet [1] architecture was chosen as it is a well known CNN for MNIST dataset.

This approach got me an accuracy result of `95%`, which is not that good but does a good job.

### Test a Model on New Images

#### Images used

Height: 425 and Width: 425
Brightness:  157.91422560553633
Mode:  RGB
Contrast:  177.98636823506638

![10 KM](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/10.jpg)

---
Height: 300 and Width: 225 <br>
Brightness:  127.69592592592592 <br>
Mode:  RGB <br>
Contrast:  145.30666763524397 <br>

![Give way](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/give_way_sign.jpg)

---
Height: 300 and Width: 225 <br>
Brightness:  127.69592592592592 <br>
Mode:  RGB <br>
Contrast:  145.30666763524397 <br>

![No entry](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/no_entry.jpg)

---
Height: 300 and Width: 225 <br>
Brightness:  93.52752592592593 <br>
Mode:  RGB <br>
Contrast:  110.50265168596466 <br>

![Stop](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/stop.jpg)

---
Height: 876 and Width: 493 <br>
Brightness:  155.05880268971075 <br>
Mode:  RGB <br>
Contrast:  167.9383344340375 <br>

![Wrong way](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/wrong_way.jpg)

These images are not German traffic signs, but they are signs from USA, UK and NZ.

Due to the difference in the shape, I had to crop and rescale the image to `(32, 32, 3)`

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

The code for making predictions on my final model is located in the cell `126` of the Jupyter notebook.

For the first image, that detected the as `No Entry`

**Image 1**

![Image 1 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image1_p.png)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lable</th>
      <th>Name</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>No passing</td>
      <td>96.5%%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>Keep left</td>
      <td>3.5%%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>No vehicles</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>Road narrows on the right</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>End of no passing by vehicles over 3.5 metric ...</td>
      <td>0.0%%</td>
    </tr>
  </tbody>
</table>

**Image 2**

![Image 2 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image2_p.png)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lable</th>
      <th>Name</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>No passing for vehicles over 3.5 metric tons</td>
      <td>99.9%%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25</td>
      <td>Road work</td>
      <td>0.1%%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>General caution</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>No entry</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>Pedestrians</td>
      <td>0.0%%</td>
    </tr>
  </tbody>
</table>

**Image 3**

![Image 3 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image3_p.png)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lable</th>
      <th>Name</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>Go straight or left</td>
      <td>100.0%%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Turn right ahead</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>Traffic signals</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>No entry</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Speed limit (50km/h)</td>
      <td>0.0%%</td>
    </tr>
  </tbody>
</table>

**Image 4**

![Image 4 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image4_p.png)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lable</th>
      <th>Name</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>Pedestrians</td>
      <td>99.2%%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Turn right ahead</td>
      <td>0.8%%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>Turn left ahead</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Speed limit (50km/h)</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15</td>
      <td>No vehicles</td>
      <td>0.0%%</td>
    </tr>
  </tbody>
</table>

**Image 5**

![Image 5 probability](https://github.com/akshaybabloo/Car-ND/raw/master/Project_2/images/image5_p.png)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lable</th>
      <th>Name</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>Right-of-way at the next intersection</td>
      <td>100.0%%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Speed limit (60km/h)</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>End of speed limit (80km/h)</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>No vehicles</td>
      <td>0.0%%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Children crossing</td>
      <td>0.0%%</td>
    </tr>
  </tbody>
</table>

## Reference

[1] LeCun, Y., Jackel, L. D., Bottou, L., Brunot, A., Cortes, C., Denker, J. S., ... & Simard, P. (1995, October). Comparison of learning algorithms for handwritten digit recognition. In International conference on artificial neural networks (Vol. 60, pp. 53-60).
