ConvNets Shapes (Dimensionality)
--------------------------------

To get the number of neurons in the next layer we have to calculate its dimensionality, whose equation is given by:

.. math::

    \frac{\left ( W-F+2P \right )}{S+1}

Where :math:`W` is the input volume, :math:`F` is the filter volume i.e. ``(height*width*depth)``, :math:`P` is number opf strides and :math:`S` is the padding.

.. automodule:: Term_1.CNN.output_shape_1
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:


Number of Parameters
--------------------

To get the memory used by the CNN, we would do the following:

.. math::

    \frac{\left ( filter~height~*~filter~width~*~filter~depth \right ) + 1}{new~layer~height~*~new~layer~width~*~new~layer~depth}

.. automodule:: Term_1.CNN.num_of_parameters_2
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

Parameter Sharing
-----------------

To get the number of parameters shared, we would do the following

.. math::

    \left ( filter~height~*~filter~width~*~filter~depth + 1 \right ) * number~of~filters + biases~or~new~layer~depth

.. automodule:: Term_1.CNN.parameter_sharing_3
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

What does each layer show?
--------------------------

A paper written by Zeiler and Fergus (2014) [1]_, shows how each layer learns.


References
----------

.. [1] Zeiler, M. D., & Fergus, R. (2014, September). Visualizing and understanding convolutional networks. In European conference on computer vision (pp. 818-833). Springer International Publishing.