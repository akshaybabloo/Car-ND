Regularization
==============

A Regularization can be done by adding artificial constraints to the network that implicitly reduces the number of free parameters. To prevent over fitting of training data in a model, we use regularization methods to stop such training to happen.

One such way to prevent over fitting is to use L2 Regularization [1]_ [2]_ whose equation is given by

.. math::

    {\zeta }'=\zeta +\beta \frac{1}{2}\left \| w \right \|_{2}^{2}

where, :math:`{\zeta }'` is the new loss, :math:`\zeta` is previous loss, :math:`\beta` is some small constant and :math:`w` is the weight.

:math:`\left \| w \right \|_{2}^{2}` can be written as,

.. math::

    \Rightarrow \left \| w \right \|_{2}^{2} \\
    \Rightarrow \left ( w_{1}^{2}+w_{2}^{2}+...+w_{n}^{2} \right )

By differentiating it with respect to ``w``, we get

.. math::

    \frac{\mathrm{d} }{\mathrm{d} w} \left ( w_{1}^{2}+w_{2}^{2}+...+w_{n}^{2} \right ) \\
    \Rightarrow w

Dropout
-------

Another important technique for regularization by Hinton (2014) [3]_, which says the following:

``IF there are two layers communicating between each other (activation data), in that take completely random data and make it equal to zero.``

This will force the date to learn redundant data to prevent over fitting. If this doesn't work then bigger network should be used.

For example, see the image below:

.. figure:: dropout-node.jpeg
   :align: center

Another option is to double the randomly unselected numbers, this will average to the original values.

.. automodule:: Term_1.DeepNeuralNetwork.dropout_8
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

Reference
---------

.. [1] https://en.wikipedia.org/wiki/Regularization_(mathematics)
.. [2] http://cs.nyu.edu/~rostami/presentations/L1_vs_L2.pdf
.. [3] Srivastava, N., Hinton, G. E., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929-1958.
