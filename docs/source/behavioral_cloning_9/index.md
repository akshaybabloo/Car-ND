Behavioral Cloning
==================

According to [1]_, behavioral cloning can be defined as a method in which the computers can imitate human skills by learning their cognitive behaviour.

In this project we will try to mimic the vehicles behaviour using CNN.

Project can be found at https://github.com/akshaybabloo/Car-ND/tree/master/Project_3


Strategy
--------

Strategies for Collecting Data
Now that you have driven the simulator and know how to record data, it's time to think about collecting data that will ensure a successful model. There are a few general concepts to think about that we will later discuss in more detail:

* the car should stay in the center of the road as much as possible
* if the car veers off to the side, it should recover back to center
* driving counter-clockwise can help the model generalize
* flipping the images is a quick way to augment the data
* collecting data from the second track can also help generalize the model
* we want to avoid over-fitting or under-fitting when training the model
* knowing when to stop collecting more data

Reference
---------

.. [1] Sammut, C. (2010). Behavioral Cloning. In C. Sammut & G. I. Webb (Eds.), Encyclopedia of Machine Learning (pp. 93-97). Boston, MA: Springer US.