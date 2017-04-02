Transfer Learning
=================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   alex_net

What is Transfer Learning?
--------------------------

Transfer learning, is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited [1]_.

You must consider using Transfer Learning when consider these four use cases:

1. new data set is small, new data is similar to original training data
2. new data set is small, new data is different from original training data
3. new data set is large, new data is similar to original training data
4. new data set is large, new data is different from original training data

A simple guide when using Transfer Learning.

.. figure:: images/02-guide-how-transfer-learning-v3-01.png
   :scale: 50%
   :align: center

   Four Cases When Using Transfer Learning (Source: Udacity).

A large data set might have one million images. A small data could have two-thousand images. The dividing line between a large data set and small data set is somewhat subjective. Overfitting is a concern when using transfer learning with a small data set.

Images of dogs and images of wolves would be considered similar; the images would share common characteristics. A data set of flower images would be different from a data set of dog images.

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

Example
-------

For example let us consider a very simple CNN network:

.. figure:: images/02-guide-how-transfer-learning-v3-02.png
   :scale: 50%
   :align: center

   A simple CNN network (Source: Udacity).

**Case 1: Small Data Set, Similar Data:**

.. figure:: images/02-guide-how-transfer-learning-v3-03.png
   :scale: 50%
   :align: center

   Case 1: Small Data Set, Similar Data (Source: Udacity).

If the new data set is small and similar to the original training data:

* slice off the end of the neural network
* add a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.

Since the data sets are similar, images from each data set will have similar higher level features. Therefore most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.

Here's how to visualize this approach:

.. figure:: images/02-guide-how-transfer-learning-v3-04.png
   :scale: 50%
   :align: center

   Neural Network with Small Data Set, Similar Data (Source: Udacity).

**Case 2: Small Data Set, Different Data:**

.. figure:: images/02-guide-how-transfer-learning-v3-05.png
   :scale: 50%
   :align: center

   Case 2: Small Data Set, Different Data (Source: Udacity).

If the new data set is small and different from the original training data:

* slice off most of the pre-trained layers near the beginning of the network
* add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
* randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
* train the network to update the weights of the new fully connected layer

Because the data set is small, overfitting is still a concern. To combat overfitting, the weights of the original neural network will be held constant, like in the first case.

But the original training set and the new data set do not share higher level features. In this case, the new network will only use the layers containing lower level features.

Here is how to visualize this approach:

.. figure:: images/02-guide-how-transfer-learning-v3-06.png
   :scale: 50%
   :align: center

   Neural Network with Small Data Set, Different Data (Source: Udacity).

**Case 3: Large Data Set, Similar Data**

.. figure:: images/02-guide-how-transfer-learning-v3-07.png
   :scale: 50%
   :align: center

   Case 3: Large Data Set, Similar Data (Source: Udacity).

If the new data set is large and similar to the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* randomly initialize the weights in the new fully connected layer
* initialize the rest of the weights using the pre-trained weights
* re-train the entire neural network

Overfitting is not as much of a concern when training on a large data set; therefore, you can re-train all of the weights.

Because the original training set and the new data set share higher level features, the entire neural network is used as well.

Here is how to visualize this approach:

.. figure:: images/02-guide-how-transfer-learning-v3-08.png
   :scale: 50%
   :align: center

   Neural Network with Large Data Set, Similar Data (Source: Udacity).

**Case 4: Large Data Set, Different Data**

.. figure:: images/02-guide-how-transfer-learning-v3-09.png
   :scale: 50%
   :align: center

   Case 4: Large Data Set, Different Data

If the new data set is large and different from the original training data:

* remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
* retrain the network from scratch with randomly initialized weights
* alternatively, you could just use the same strategy as the "large and similar" data case

Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set.

If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

Here is how to visualize this approach:

.. figure:: images/02-guide-how-transfer-learning-v3-10.png
   :scale: 50%
   :align: center

   Neural Network with Large Data Set, Different Data (Source: Udacity).

Reference
---------

.. [1] https://en.wikipedia.org/wiki/Inductive_transfer "Inductive transfer"
