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

    \frac{\left ( filter~height~*~filter~width~*~filter~depth \right ) + 1{new~layer~height~*~new~layer~width~*~new~layer~depth}

.. automodule:: Term_1.CNN.num_of_parameters_2
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members: