AlexNet
=======

AlexNet puts the network on two GPUs, which allows for building a larger network. Although most of the calculations are done in parallel, the GPUs communicate with each other in certain layers. The original research paper on AlexNet said that parallelizing the network decreased the classification error rate by 1.7% when compared to a neural network that used half as many neurons on one GPU.

.. figure:: images/08-alexnet-1.png
   :scale: 30%
   :align: center

   AlexNet architecture (Source: Udacity).