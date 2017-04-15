AlexNet
=======

AlexNet [1]_ puts the network on two GPUs, which allows for building a larger network. Although most of the calculations are done in parallel, the GPUs communicate with each other in certain layers. The original research paper on AlexNet said that parallelizing the network decreased the classification error rate by 1.7% when compared to a neural network that used half as many neurons on one GPU.

For code see https://github.com/akshaybabloo/Car-ND/tree/master/Term_1/Transfer_Learning_8/Alexnet_8_1

.. figure:: images/08-alexnet-1.png
   :scale: 30%
   :align: center

   AlexNet architecture (Source: Udacity).

Reference
---------

.. [1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).