Sobel Operator
==============

The Sobel operator is at the heart of the Canny edge detection algorithm you used in the Introductory Lesson. Applying the Sobel operator to an image is a way of taking the derivative of the image in the :math:`x` or :math:`y` direction. The operators for :math:`sobel_{x}` and :math:`sobel_{y}`  respectively, look like this:

.. math::

    S_{x} = \begin{bmatrix}
            -1 & 0 & 1\\
            -2 & 0 & 2\\
            -1 & 0 & 1
            \end{bmatrix} \\
    S_{y} = \begin{bmatrix}
            -1 & -2 & -1\\
            0 & 0 & 0\\
            1 & 2 & 1
            \end{bmatrix}

These are examples of Sobel operators with a kernel size of 3 (implying a 3 x 3 operator in each case). This is the minimum size, but the kernel size can be any odd number. A larger kernel implies taking the gradient over a larger region of the image, or, in other words, a smoother gradient.

To understand how these operators take the derivative, you can think of overlaying either one on a 3 x 3 region of an image. If the image is flat across that region, then the result (summing the element-wise product of the operator and corresponding image pixels) will be zero. If, instead, for example, you apply the :math:`S_{x}` operator to a region of the image where values are rising from left to right, then the result will be positive, implying a positive derivative.

Derivative Example
------------------

If we apply the Sobel x and y operators to this image:

.. figure:: curved-lane.jpg
   :align: center

   Source: Udacity

And then we take the absolute value, we get the result:

.. figure:: screen-shot-2016-12-01-at-4.50.36-pm.png
   :align: center

   Source: Udacity