Transfer Learning
=================

Transfer learning, is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited [1]_.

You must consider using Transfer Learning when consider these four use cases:

1. new data set is small, new data is similar to original training data
2. new data set is small, new data is different from original training data
3. new data set is large, new data is similar to original training data
4. new data set is large, new data is different from original training data

A simple guide when using Transfer Learning.

.. figure:: 02-guide-how-transfer-learning-v3-01.png
   :align: center

   Four Cases When Using Transfer Learning (Source: Udacity).

A large data set might have one million images. A small data could have two-thousand images. The dividing line between a large data set and small data set is somewhat subjective. Overfitting is a concern when using transfer learning with a small data set.

Images of dogs and images of wolves would be considered similar; the images would share common characteristics. A data set of flower images would be different from a data set of dog images.

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

Example
-------

For example let us consider a very simple CNN network:

.. figure:: 02-guide-how-transfer-learning-v3-02.png
   :align: center

   A simple CNN network (Source: Udacity).

**Case 1: Small Data Set, Similar Data:**

.. figure:: 02-guide-how-transfer-learning-v3-03.png
   :align: center

   Small Data Set, Similar Data (Source: Udacity).


Reference
---------

.. [1] https://en.wikipedia.org/wiki/Inductive_transfer "Inductive transfer"
